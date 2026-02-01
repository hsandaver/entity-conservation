"""RDF parsing, SPARQL, NetworkX, embeddings logic."""

from __future__ import annotations

import difflib
import io
import hashlib
import json
import logging
import os
import re
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
import streamlit as st
from rdflib import ConjunctiveGraph, Graph as RDFGraph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from src.config import ASSERTED_CTX, CONFIG, EX, INFERRED_CTX
from src.models import Edge, GraphData, Node
from src.utils import (
    _label_is_unnamed,
    _local_name,
    _predicate_uri,
    _type_uri,
    canonical_type,
    normalize_data,
    normalize_relationship_value,
    profile_time,
    remove_fragment,
    validate_entity,
)

try:
    from pyshacl import validate

    pyshacl_installed = True
except ImportError:
    pyshacl_installed = False

try:
    from owlrl import DeductiveClosure, RDFS_Semantics

    owlrl_installed = True
except ImportError:
    owlrl_installed = False

try:
    from node2vec import Node2Vec

    node2vec_installed = True
except ImportError:
    node2vec_installed = False

try:
    from sentence_transformers import SentenceTransformer

    sentence_transformer_installed = True
except ImportError:
    sentence_transformer_installed = False

try:
    from pymarc import MARCReader

    pymarc_installed = True
except ImportError:
    pymarc_installed = False


# ------------------------------
# OCLC Entity Query Auth & SPARQL Helpers
# ------------------------------
_oclc_token_cache = {"access_token": None, "expires_at": 0.0}


@profile_time
def get_entity_query_token(skew: int = 60) -> str:
    """
    Get a fresh access token for the OCLC Entity Query API using Client Credentials.
    Tokens are short-lived; we cache in memory and renew with `skew` seconds of headroom.
    Looks in Streamlit secrets first, then environment:
      OCLC_CLIENT_ID (WSKey) / OCLC_CLIENT_SECRET
    """
    wskey = None
    secret = None
    try:
        wskey = st.secrets.get("OCLC_CLIENT_ID", None)
        secret = st.secrets.get("OCLC_CLIENT_SECRET", None)
    except Exception:
        pass
    if not wskey:
        wskey = os.getenv("OCLC_CLIENT_ID")
    if not secret:
        secret = os.getenv("OCLC_CLIENT_SECRET")

    if not wskey or not secret:
        raise RuntimeError(
            "Missing OCLC credentials. Set OCLC_CLIENT_ID and OCLC_CLIENT_SECRET in Streamlit secrets or env."
        )

    now = time.time()
    tok = _oclc_token_cache.get("access_token")
    exp = float(_oclc_token_cache.get("expires_at", 0.0))
    if tok and now < (exp - skew):
        return tok

    resp = requests.post(
        "https://oauth.oclc.org/token",
        auth=(wskey, secret),
        headers={"Accept": "application/json"},
        data={"grant_type": "client_credentials", "scope": "entity-query"},
        timeout=30,
    )
    try:
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"OCLC token request failed: {exc}\nBody: {resp.text[:500]}")

    data = resp.json()
    token = data["access_token"]
    expires_in = int(data.get("expires_in", 900))
    _oclc_token_cache["access_token"] = token
    _oclc_token_cache["expires_at"] = now + expires_in
    return token


@profile_time
def oclc_sparql(query: str, result_type: str = "select") -> Any:
    """
    Run a SPARQL query against the OCLC Entity Query endpoint using a Bearer token.
    result_type: "select" -> JSON bindings; "graph" -> JSON-LD (for CONSTRUCT/DESCRIBE)
    Uses GET with the 'query' parameter for compatibility with the API.
    """
    token = get_entity_query_token()
    endpoint = "https://entities.api.oclc.org/v1/sparql"
    accept = "application/ld+json" if result_type.lower() == "graph" else "application/sparql-results+json"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
    }

    resp = requests.get(endpoint, params={"query": query}, headers=headers, timeout=60)

    if resp.status_code in (401, 403):
        _oclc_token_cache["access_token"] = None
        _oclc_token_cache["expires_at"] = 0.0
        token = get_entity_query_token()
        headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(endpoint, params={"query": query}, headers=headers, timeout=60)

    try:
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"SPARQL request failed: {exc}\nBody: {resp.text[:1000]}")

    return resp.json()

# ------------------------------
# RDF Ingestion and Validation
# ------------------------------

def parse_rdf_data(content: str, file_extension: str) -> RDFGraph:
    rdf_graph = RDFGraph()
    fmt = None
    if file_extension == "ttl":
        fmt = "turtle"
    elif file_extension == "rdf":
        fmt = "xml"
    elif file_extension == "nt":
        fmt = "nt"
    else:
        raise ValueError("Unsupported RDF format.")
    rdf_graph.parse(data=content, format=fmt)
    return rdf_graph


def process_uploaded_file(uploaded_file) -> Tuple[List[RDFGraph], List[str]]:
    graphs = []
    errors = []
    try:
        content = uploaded_file.read().decode("utf-8")
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["json", "jsonld"]:
            json_obj = json.loads(content)
            g = RDFGraph().parse(data=json.dumps(json_obj), format="json-ld")
            graphs.append(g)
        elif file_extension in ["ttl", "rdf", "nt"]:
            g = parse_rdf_data(content, file_extension)
            graphs.append(g)
        else:
            errors.append(f"Unsupported file format: {uploaded_file.name}")
    except Exception as exc:
        errors.append(f"Error processing file {uploaded_file.name}: {exc}")
    return graphs, errors


# ------------------------------
# RIS Ingestion + Fuzzy Matching
# ------------------------------

_RIS_TAG_RE = re.compile(r"^([A-Z0-9]{2})  - ?(.*)$")
_RIS_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_WORK_MERGE_THRESHOLD = 0.92
_WORK_LINK_THRESHOLD = 0.85
_PERSON_MERGE_THRESHOLD = 0.93
_PERSON_LINK_THRESHOLD = 0.88
_ORG_MERGE_THRESHOLD = 0.92
_ORG_LINK_THRESHOLD = 0.87
_AMBIGUOUS_GAP = 0.02

_LINKED_ART_RELATIONSHIP_MAP = {
    "1101": ("studentOf", True),  # teacher of -> studentOf (reverse)
    "1102": ("studentOf", False),
    "1217": ("employedBy", False),
    "1218": ("employedBy", True),  # employee was -> employedBy (reverse)
    "1305": ("associatedWith", False),
    "1311": ("associatedWith", False),
    "1317": ("memberOf", False),
    "1511": ("child", False),
    "1512": ("child", True),  # parent of -> child (reverse)
    "1513": ("relatedPerson", False),
    "1514": ("relatedPerson", False),
    "1516": ("relatedPerson", False),
    "1531": ("relatedPerson", False),
    "1543": ("spouse", True),  # consort was -> spouse (reverse)
    "1550": ("relatedPerson", False),
}

_LINKED_ART_LABEL_HINTS = [
    ("teacher of", "studentOf", True),
    ("student of", "studentOf", False),
    ("employee of", "employedBy", False),
    ("employee was", "employedBy", True),
    ("worked with", "associatedWith", False),
    ("partner of", "associatedWith", False),
    ("member of", "memberOf", False),
    ("child of", "child", False),
    ("parent of", "child", True),
    ("grandchild of", "relatedPerson", False),
    ("grandparent of", "relatedPerson", False),
    ("nephew", "relatedPerson", False),
    ("niece", "relatedPerson", False),
    ("consort was", "spouse", True),
    ("relative by marriage", "relatedPerson", False),
]

_OCLC_ENTITY_PREFIX = "https://id.oclc.org/worldcat/entity/"
_ROLE_PERSON_RELATIONSHIPS = {
    "author",
    "contributor",
    "creator",
    "draftsman",
    "artist",
    "editor",
    "designer",
    "composer",
    "translator",
    "illustrator",
}
_OCLC_ENTITY_CACHE: Dict[str, Dict[str, Any]] = {}


def collapse_same_as_nodes(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    hide_prefixes: Optional[List[str]] = None,
) -> Tuple[GraphData, Dict[str, str], Dict[str, int]]:
    if not graph_data.nodes:
        return graph_data, id_to_label, {"aliases": 0, "rewired": 0}
    hide_prefixes = hide_prefixes or [_OCLC_ENTITY_PREFIX]
    existing_ids = {node.id for node in graph_data.nodes if node.id}
    alias_map: Dict[str, str] = {}
    for node in graph_data.nodes:
        node_id = node.id
        if not isinstance(node_id, str):
            continue
        if not any(node_id.startswith(prefix) for prefix in hide_prefixes):
            continue
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        same_as_list = _normalize_same_as(meta.get("sameAs"))
        if not same_as_list:
            continue
        target_id = None
        for candidate in same_as_list:
            if candidate in existing_ids and not any(
                candidate.startswith(prefix) for prefix in hide_prefixes
            ):
                target_id = candidate
                break
        if not target_id:
            for candidate in same_as_list:
                if candidate in existing_ids:
                    target_id = candidate
                    break
        if target_id and target_id != node_id:
            alias_map[node_id] = target_id

    if not alias_map:
        return graph_data, id_to_label, {"aliases": 0, "rewired": 0}

    node_by_id = {node.id: node for node in graph_data.nodes}
    for alias_id, target_id in alias_map.items():
        target_node = node_by_id.get(target_id)
        if not target_node or not isinstance(target_node.metadata, dict):
            continue
        same_as_values = _normalize_same_as(target_node.metadata.get("sameAs"))
        if alias_id not in same_as_values:
            same_as_values.append(alias_id)
            target_node.metadata["sameAs"] = same_as_values

    edges_by_source: Dict[str, List[Edge]] = {}
    edge_keys: Set[Tuple[str, str, str]] = set()
    rewired = 0
    for node in graph_data.nodes:
        for edge in node.edges:
            src = alias_map.get(edge.source, edge.source)
            dst = alias_map.get(edge.target, edge.target)
            if src != edge.source or dst != edge.target:
                rewired += 1
            if src == dst:
                continue
            key = (src, dst, edge.relationship)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges_by_source.setdefault(src, []).append(
                Edge(
                    source=src,
                    target=dst,
                    relationship=edge.relationship,
                    inferred=getattr(edge, "inferred", False),
                    weight=getattr(edge, "weight", None),
                )
            )

    kept_nodes: List[Node] = []
    for node in graph_data.nodes:
        if node.id in alias_map:
            continue
        node.edges = edges_by_source.get(node.id, [])
        kept_nodes.append(node)

    cleaned_labels = {node.id: id_to_label.get(node.id, node.label) for node in kept_nodes}
    return GraphData(nodes=kept_nodes), cleaned_labels, {"aliases": len(alias_map), "rewired": rewired}


def _normalize_ris_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(text: str, drop_stopwords: bool = True) -> Set[str]:
    if not text:
        return set()
    tokens = text.split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _RIS_STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return set(tokens)


def _similarity(norm_a: str, tokens_a: Set[str], norm_b: str, tokens_b: Set[str]) -> float:
    if not norm_a or not norm_b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    if tokens_a and tokens_b:
        jaccard = len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))
    else:
        jaccard = 0.0
    score = 0.6 * ratio + 0.4 * jaccard
    if norm_a in norm_b or norm_b in norm_a:
        score = min(1.0, score + 0.05)
    return score


def _best_fuzzy_match(
    label: str,
    candidates: List[Tuple[str, str, Set[str]]],
    drop_stopwords: bool,
    min_score: float,
) -> Tuple[Optional[str], float, bool]:
    norm = _normalize_ris_text(label)
    tokens = _tokenize(norm, drop_stopwords=drop_stopwords)
    best_id = None
    best_score = 0.0
    second_score = 0.0
    for cand_id, cand_norm, cand_tokens in candidates:
        score = _similarity(norm, tokens, cand_norm, cand_tokens)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_id = cand_id
        elif score > second_score:
            second_score = score
    if best_score < min_score:
        return None, best_score, False
    if best_score - second_score < _AMBIGUOUS_GAP:
        return None, best_score, True
    return best_id, best_score, False


def _extract_linked_art_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"(\\d+)$", value)
    return match.group(1) if match else None


def _map_linked_art_relationship(rel: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    classified = rel.get("classified_as", [])
    if isinstance(classified, dict):
        classified = [classified]
    labels: List[str] = []
    if isinstance(classified, list):
        for entry in classified:
            cid = None
            label = None
            if isinstance(entry, dict):
                cid = entry.get("id")
                label = entry.get("_label") or entry.get("label") or entry.get("content")
            elif isinstance(entry, str):
                cid = entry
            code = _extract_linked_art_code(cid) if cid else None
            if code and code in _LINKED_ART_RELATIONSHIP_MAP:
                return _LINKED_ART_RELATIONSHIP_MAP[code]
            if isinstance(label, str) and label.strip():
                labels.append(label.strip().lower())
            if isinstance(cid, str):
                labels.append(cid.strip().lower())
    for label in labels:
        for needle, rel_key, reverse in _LINKED_ART_LABEL_HINTS:
            if needle in label:
                return rel_key, reverse
    return None, False


_EDGE_WEIGHT_KEYS = ("confidence", "score", "weight", "probability", "certainty", "likelihood")


def _extract_edge_weight(value: Any) -> Optional[float]:
    if not isinstance(value, dict):
        return None
    for key in _EDGE_WEIGHT_KEYS:
        if key in value:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_linked_art_edges(data: Dict[str, Any], subject_id: str) -> List[Edge]:
    related = data.get("la:related_from_by") or data.get("related_from_by")
    if not related:
        return []
    if isinstance(related, dict):
        related = [related]
    if not isinstance(related, list):
        return []
    edges: List[Edge] = []
    for rel in related:
        if not isinstance(rel, dict):
            continue
        weight = _extract_edge_weight(rel)
        target_ref = rel.get("la:relates_to") or rel.get("relates_to") or rel.get("related_to")
        target_id = None
        if isinstance(target_ref, dict):
            target_id = target_ref.get("id") or target_ref.get("@id")
        elif isinstance(target_ref, str):
            target_id = target_ref
        if not target_id:
            continue
        target_id = remove_fragment(target_id)
        rel_key, reverse = _map_linked_art_relationship(rel)
        if not rel_key:
            rel_key = "relatedPerson"
        if reverse:
            edges.append(
                Edge(source=target_id, target=subject_id, relationship=rel_key, weight=weight)
            )
        else:
            edges.append(
                Edge(source=subject_id, target=target_id, relationship=rel_key, weight=weight)
            )
    seen: Set[Tuple[str, str, str]] = set()
    unique_edges: List[Edge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relationship)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(edge)
    return unique_edges


def _extract_entity_label(entity: Dict[str, Any]) -> Optional[str]:
    pref = entity.get("prefLabel")
    if isinstance(pref, dict):
        for key in ("en", "la"):
            raw = pref.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        for val in pref.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    elif isinstance(pref, str) and pref.strip():
        return pref.strip()
    for key in ("_label", "label", "content", "name"):
        raw = entity.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _normalize_same_as(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [normalize_relationship_value("sameAs", value) or value]
    if isinstance(value, list):
        normalized: List[str] = []
        for v in value:
            if not isinstance(v, str):
                continue
            normalized.append(normalize_relationship_value("sameAs", v) or v)
        return normalized
    return []


def _fetch_oclc_entity(entity_id: str) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    cached = _OCLC_ENTITY_CACHE.get(entity_id)
    if cached is not None:
        return cached.get("data"), False, bool(cached.get("ok"))
    try:
        resp = requests.get(
            entity_id,
            headers={"Accept": "application/ld+json, application/json"},
            timeout=10,
        )
        ok = resp.status_code == 200
        data = resp.json() if ok else None
        _OCLC_ENTITY_CACHE[entity_id] = {"data": data, "ok": ok}
        return data, True, ok
    except Exception as exc:
        logging.warning("Failed to resolve OCLC entity %s: %s", entity_id, exc)
        _OCLC_ENTITY_CACHE[entity_id] = {"data": None, "ok": False}
        return None, True, False


def link_oclc_creators(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    max_fetch: int = 12,
) -> Tuple[GraphData, Dict[str, str], Dict[str, int]]:
    if not graph_data.nodes:
        return graph_data, id_to_label, {"targets": 0, "resolved": 0, "mapped": 0, "fetched": 0, "fetch_failed": 0}

    same_as_index: Dict[str, str] = {}
    for node in graph_data.nodes:
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        for same_as in _normalize_same_as(meta.get("sameAs")):
            if same_as not in same_as_index:
                same_as_index[same_as] = node.id

    person_candidates = _build_fuzzy_index(graph_data, id_to_label)
    person_pool = person_candidates.get("Person", []) + person_candidates.get("Unknown", [])

    oclc_targets: Set[str] = set()
    for node in graph_data.nodes:
        for edge in node.edges:
            if edge.relationship not in _ROLE_PERSON_RELATIONSHIPS:
                continue
            target_id = edge.target
            if isinstance(target_id, str) and target_id.startswith(_OCLC_ENTITY_PREFIX):
                oclc_targets.add(target_id)

    external_to_internal: Dict[str, str] = {}
    targets = len(oclc_targets)
    fetched = 0
    fetch_failed = 0
    resolved = 0
    mapped = 0

    for target_id in sorted(oclc_targets):
        if target_id in external_to_internal:
            continue
        mapped_id = same_as_index.get(target_id)
        label = id_to_label.get(target_id)
        same_as_list: List[str] = []
        if not mapped_id and fetched < max_fetch:
            entity, did_fetch, fetch_ok = _fetch_oclc_entity(target_id)
            if did_fetch:
                fetched += 1
                if not fetch_ok:
                    fetch_failed += 1
            if entity:
                label = label if label and not _label_is_unnamed(label, target_id) else _extract_entity_label(entity)
                same_as_list = _normalize_same_as(entity.get("sameAs"))
                for same_as in same_as_list:
                    mapped_id = same_as_index.get(same_as)
                    if mapped_id:
                        break
        if not mapped_id and label:
            match_id, score, ambiguous = _best_fuzzy_match(
                label,
                person_pool,
                drop_stopwords=False,
                min_score=_PERSON_LINK_THRESHOLD,
            )
            if match_id and not ambiguous:
                mapped_id = match_id
        if mapped_id:
            external_to_internal[target_id] = mapped_id
            resolved += 1

    if not external_to_internal:
        return graph_data, id_to_label, {
            "targets": targets,
            "resolved": 0,
            "mapped": 0,
            "fetched": fetched,
            "fetch_failed": fetch_failed,
        }

    for node in graph_data.nodes:
        for edge in node.edges:
            if edge.relationship not in _ROLE_PERSON_RELATIONSHIPS:
                continue
            mapped_id = external_to_internal.get(edge.target)
            if mapped_id and mapped_id != edge.target:
                edge.target = mapped_id
                mapped += 1

    return graph_data, id_to_label, {
        "targets": targets,
        "resolved": resolved,
        "mapped": mapped,
        "fetched": fetched,
        "fetch_failed": fetch_failed,
    }


def _normalize_person_name(name: str) -> str:
    cleaned = " ".join(name.replace(".", " ").split())
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            cleaned = f"{parts[1]} {parts[0]}"
    return cleaned.strip()


def _normalize_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("/")
    return text.lower() if text else None


def _doi_to_url(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def _normalize_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = raw.strip()
    text = text.replace(" ", "")
    if not text:
        return None
    return remove_fragment(text)


def _extract_year(values: List[str]) -> Optional[str]:
    for value in values:
        match = re.search(r"(\d{4})", value)
        if match:
            return match.group(1)
    return None


def _parse_ris_records(content: str) -> List[Dict[str, List[str]]]:
    records: List[Dict[str, List[str]]] = []
    record: Dict[str, List[str]] = {}
    last_tag: Optional[str] = None
    for raw_line in content.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        match = _RIS_TAG_RE.match(line)
        if match:
            tag = match.group(1)
            value = match.group(2).strip()
            if tag == "ER":
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            last_tag = tag
            record.setdefault(tag, [])
            if value:
                record[tag].append(value)
            continue
        if last_tag:
            continuation = line.strip()
            if continuation:
                if record.get(last_tag):
                    record[last_tag][-1] = f"{record[last_tag][-1]} {continuation}"
                else:
                    record[last_tag] = [continuation]
    if record:
        records.append(record)
    return records


def _make_ris_id(prefix: str, parts: List[str]) -> str:
    seed = "|".join(p for p in parts if p)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"urn:ris:{prefix}:{digest}"


def _build_fuzzy_index(
    graph_data: Optional[GraphData],
    id_to_label: Optional[Dict[str, str]],
) -> Dict[str, List[Tuple[str, str, Set[str]]]]:
    index: Dict[str, List[Tuple[str, str, Set[str]]]] = {
        "Person": [],
        "Organization": [],
        "Work": [],
        "Place": [],
        "Unknown": [],
    }
    if not graph_data or not graph_data.nodes:
        return index
    label_map = id_to_label or {}
    for node in graph_data.nodes:
        label = label_map.get(node.id) or node.label or _local_name(node.id)
        norm = _normalize_ris_text(label)
        if not norm:
            continue
        node_types = [canonical_type(t) for t in (node.types or [])] or ["Unknown"]
        primary_type = node_types[0] if node_types else "Unknown"
        drop_stopwords = primary_type != "Person"
        tokens = _tokenize(norm, drop_stopwords=drop_stopwords)
        bucket = index.get(primary_type, index["Unknown"])
        bucket.append((node.id, norm, tokens))
    return index


def _add_candidate(
    index: Dict[str, List[Tuple[str, str, Set[str]]]],
    node_id: str,
    label: str,
    node_type: str,
) -> None:
    norm = _normalize_ris_text(label)
    if not norm:
        return
    drop_stopwords = node_type != "Person"
    tokens = _tokenize(norm, drop_stopwords=drop_stopwords)
    bucket = index.get(node_type, index["Unknown"])
    bucket.append((node_id, norm, tokens))


def _build_lookup_maps(graph_data: Optional[GraphData]) -> Tuple[Dict[str, str], Dict[str, str]]:
    doi_lookup: Dict[str, str] = {}
    url_lookup: Dict[str, str] = {}
    if not graph_data:
        return doi_lookup, url_lookup
    for node in graph_data.nodes:
        node_id = node.id
        doi = _normalize_doi(node_id)
        if doi:
            doi_lookup.setdefault(doi, node_id)
        if isinstance(node_id, str) and "doi.org/" in node_id.lower():
            extracted = _normalize_doi(node_id)
            if extracted:
                doi_lookup.setdefault(extracted, node_id)
        url = _normalize_url(node_id)
        if url:
            url_lookup.setdefault(url, node_id)
    return doi_lookup, url_lookup


def parse_ris_content(
    content: str,
    base_graph: Optional[GraphData] = None,
    base_id_to_label: Optional[Dict[str, str]] = None,
) -> Tuple[GraphData, Dict[str, str], List[str], Dict[str, int]]:
    records = _parse_ris_records(content)
    stats = {
        "records": 0,
        "works_created": 0,
        "works_direct_matched": 0,
        "works_fuzzy_merged": 0,
        "works_fuzzy_linked": 0,
        "works_ambiguous": 0,
        "persons_created": 0,
        "persons_merged": 0,
        "persons_linked": 0,
        "orgs_created": 0,
        "orgs_merged": 0,
        "orgs_linked": 0,
        "edges_created": 0,
    }
    errors: List[str] = []
    node_map: Dict[str, Node] = {}
    id_to_label: Dict[str, str] = {}

    base_graph = base_graph or GraphData(nodes=[])
    base_id_to_label = base_id_to_label or {}
    index = _build_fuzzy_index(base_graph, base_id_to_label)
    doi_lookup, url_lookup = _build_lookup_maps(base_graph)

    person_cache: Dict[str, str] = {}
    org_cache: Dict[str, str] = {}

    def upsert_node(node: Node) -> None:
        existing = node_map.get(node.id)
        if existing is None:
            node_map[node.id] = node
            return
        if not existing.label and node.label:
            existing.label = node.label
        existing_types = list(existing.types or [])
        for node_type in node.types or []:
            if node_type not in existing_types:
                existing_types.append(node_type)
        existing.types = existing_types
        if isinstance(existing.metadata, dict) and isinstance(node.metadata, dict):
            for key, value in node.metadata.items():
                if key not in existing.metadata:
                    existing.metadata[key] = value
        elif not existing.metadata and node.metadata:
            existing.metadata = node.metadata
        existing_edge_keys = {(e.source, e.target, e.relationship) for e in existing.edges}
        for edge in node.edges:
            key = (edge.source, edge.target, edge.relationship)
            if key not in existing_edge_keys:
                existing.edges.append(edge)
                existing_edge_keys.add(key)

    def resolve_entity(
        label: str,
        node_type: str,
        merge_threshold: float,
        link_threshold: float,
        cache: Dict[str, str],
    ) -> str:
        normalized_label = _normalize_ris_text(label)
        if normalized_label in cache:
            return cache[normalized_label]

        candidates = index.get(node_type, [])
        if node_type != "Person":
            candidates = candidates + index.get("Unknown", [])
        match_id, score, ambiguous = _best_fuzzy_match(
            label,
            candidates,
            drop_stopwords=node_type != "Person",
            min_score=link_threshold,
        )
        if match_id and not ambiguous:
            if score >= merge_threshold:
                cache[normalized_label] = match_id
                if node_type == "Person":
                    stats["persons_merged"] += 1
                elif node_type == "Organization":
                    stats["orgs_merged"] += 1
                return match_id
            new_id = _make_ris_id(node_type.lower(), [normalized_label])
            same_as_edge = Edge(
                source=new_id,
                target=match_id,
                relationship="sameAs",
                weight=score,
            )
            node = Node(
                id=new_id,
                label=label,
                types=[node_type],
                metadata={"id": new_id, "prefLabel": {"en": label}, "type": [node_type]},
                edges=[same_as_edge],
            )
            upsert_node(node)
            id_to_label[new_id] = label
            cache[normalized_label] = new_id
            _add_candidate(index, new_id, label, node_type)
            if node_type == "Person":
                stats["persons_linked"] += 1
            elif node_type == "Organization":
                stats["orgs_linked"] += 1
            stats["edges_created"] += 1
            return new_id

        new_id = _make_ris_id(node_type.lower(), [normalized_label])
        node = Node(
            id=new_id,
            label=label,
            types=[node_type],
            metadata={"id": new_id, "prefLabel": {"en": label}, "type": [node_type]},
            edges=[],
        )
        upsert_node(node)
        id_to_label[new_id] = label
        cache[normalized_label] = new_id
        _add_candidate(index, new_id, label, node_type)
        if node_type == "Person":
            stats["persons_created"] += 1
        elif node_type == "Organization":
            stats["orgs_created"] += 1
        return new_id

    for record in records:
        stats["records"] += 1
        try:
            title = (record.get("TI") or record.get("T1") or record.get("CT") or [""])[0].strip()
            if not title:
                title = (record.get("BT") or [""])[0].strip()
            authors_raw = (
                record.get("AU", [])
                + record.get("A1", [])
                + record.get("A2", [])
                + record.get("A3", [])
            )
            editors_raw = record.get("ED", [])
            journal = (record.get("T2") or record.get("JF") or record.get("JO") or record.get("JA") or [""])
            journal_title = journal[0].strip() if journal else ""
            year = _extract_year(record.get("PY", []) + record.get("Y1", []) + record.get("DA", []))
            doi = _normalize_doi((record.get("DO") or [""])[0])
            doi_url = _doi_to_url(doi)
            url = _normalize_url((record.get("UR") or [""])[0])
            keywords = [kw.strip() for kw in record.get("KW", []) if kw.strip()]
            abstract = (record.get("AB") or [""])[0].strip()
            volume = (record.get("VL") or [""])[0].strip()
            issue = (record.get("IS") or [""])[0].strip()
            start_page = (record.get("SP") or [""])[0].strip()
            end_page = (record.get("EP") or [""])[0].strip()
            record_type = (record.get("TY") or [""])[0].strip()

            work_id = doi_url or url or _make_ris_id("work", [title, year or "", ";".join(authors_raw)])
            same_as_target: Optional[str] = None
            same_as_weight: Optional[float] = None
            direct_match = None
            if doi and doi in doi_lookup:
                direct_match = doi_lookup[doi]
            elif url and url in url_lookup:
                direct_match = url_lookup[url]
            if direct_match:
                work_id = direct_match
                stats["works_direct_matched"] += 1
            elif title:
                candidates = index.get("Work", []) + index.get("Unknown", [])
                match_id, score, ambiguous = _best_fuzzy_match(
                    title,
                    candidates,
                    drop_stopwords=True,
                    min_score=_WORK_LINK_THRESHOLD,
                )
                if match_id and not ambiguous:
                    if score >= _WORK_MERGE_THRESHOLD:
                        work_id = match_id
                        stats["works_fuzzy_merged"] += 1
                    else:
                        same_as_target = match_id
                        same_as_weight = score
                        stats["works_fuzzy_linked"] += 1
                elif ambiguous:
                    stats["works_ambiguous"] += 1

            work_metadata: Dict[str, Any] = {
                "id": work_id,
                "prefLabel": {"en": title or work_id},
                "type": ["Work"],
                "source": "RIS",
            }
            if record_type:
                work_metadata["recordType"] = record_type
            if year:
                work_metadata["year"] = year
            if doi:
                work_metadata["doi"] = doi
            if url:
                work_metadata["url"] = url
            if journal_title:
                work_metadata["publicationTitle"] = journal_title
            if volume:
                work_metadata["volume"] = volume
            if issue:
                work_metadata["issue"] = issue
            if start_page:
                work_metadata["startPage"] = start_page
            if end_page:
                work_metadata["endPage"] = end_page
            if keywords:
                work_metadata["keywords"] = keywords
            if abstract:
                work_metadata["abstract"] = abstract

            work_node = Node(
                id=work_id,
                label=title or work_id,
                types=["Work"],
                metadata=work_metadata,
                edges=[],
            )
            if same_as_target and same_as_target != work_id:
                work_node.edges.append(
                    Edge(
                        source=work_id,
                        target=same_as_target,
                        relationship="sameAs",
                        weight=same_as_weight,
                    )
                )
                stats["edges_created"] += 1

            author_names = [
                _normalize_person_name(name)
                for name in authors_raw
                if name and _normalize_person_name(name)
            ]
            if author_names:
                work_metadata["authors"] = author_names
            for author in author_names:
                person_id = resolve_entity(
                    author,
                    "Person",
                    _PERSON_MERGE_THRESHOLD,
                    _PERSON_LINK_THRESHOLD,
                    person_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=person_id, relationship="author"))
                stats["edges_created"] += 1

            editor_names = [
                _normalize_person_name(name)
                for name in editors_raw
                if name and _normalize_person_name(name)
            ]
            if editor_names:
                work_metadata["editors"] = editor_names
            for editor in editor_names:
                person_id = resolve_entity(
                    editor,
                    "Person",
                    _PERSON_MERGE_THRESHOLD,
                    _PERSON_LINK_THRESHOLD,
                    person_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=person_id, relationship="editor"))
                stats["edges_created"] += 1

            if journal_title:
                org_id = resolve_entity(
                    journal_title,
                    "Organization",
                    _ORG_MERGE_THRESHOLD,
                    _ORG_LINK_THRESHOLD,
                    org_cache,
                )
                work_node.edges.append(
                    Edge(source=work_id, target=org_id, relationship="relatedOrganization")
                )
                stats["edges_created"] += 1

            upsert_node(work_node)
            id_to_label.setdefault(work_id, work_node.label)
            if work_id not in base_id_to_label:
                _add_candidate(index, work_id, work_node.label, "Work")
            stats["works_created"] += 1
            if doi:
                doi_lookup.setdefault(doi, work_id)
            if url:
                url_lookup.setdefault(url, work_id)
        except Exception as exc:
            errors.append(f"RIS record {stats['records']}: {exc}")

    return GraphData(nodes=list(node_map.values())), id_to_label, errors, stats


def _make_marc_id(prefix: str, parts: List[str]) -> str:
    seed = "|".join(str(p) for p in parts if p)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"urn:marc:{prefix}:{digest}"


def _clean_marc_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = " ".join(str(value).split())
    text = re.sub(r"[\\s\\/:;,]+$", "", text)
    return text.strip()


def _strip_paren_suffix(value: str) -> str:
    if not value:
        return value
    return re.split(r"\\s*\\(", value, maxsplit=1)[0].strip()


def _coerce_marc_list(values: List[str]) -> List[str]:
    cleaned = []
    for value in values:
        text = _clean_marc_text(value)
        if text:
            cleaned.append(text)
    return cleaned


def _format_marc_name(field) -> str:
    if not field:
        return ""
    parts: List[str] = []
    for code in ("a", "b", "c", "d", "q", "n", "p"):
        parts.extend(field.get_subfields(code))
    if not parts:
        value = field.value() or ""
        return _clean_marc_text(value)
    text = " ".join(parts)
    return _clean_marc_text(text)


def _collect_marc_notes(record) -> List[Dict[str, str]]:
    notes: List[Dict[str, str]] = []
    for tag, label in (("500", "Note"), ("502", "Dissertation Note"), ("504", "Bibliography Note")):
        for field in record.get_fields(tag):
            content = _clean_marc_text(" ".join(field.get_subfields("a", "b", "c")))
            if content:
                notes.append({"classified_as": label, "content": content})
    return notes


def _parse_marc_583(field) -> Dict[str, Any]:
    if not field:
        return {}

    def _sub(code: str) -> List[str]:
        return _coerce_marc_list(field.get_subfields(code))

    action = _sub("a")
    action_id = _sub("b")
    date = _sub("c")
    interval = _sub("d")
    contingency = _sub("e")
    authorization = _sub("f")
    jurisdiction = _sub("h")
    method = _sub("i")
    site = _sub("j")
    agent = _sub("k")
    status = _sub("l")
    extent = _sub("n")
    unit_type = _sub("o")
    uri = _sub("u")
    nonpublic = _sub("x")
    public = _sub("z")
    source = _sub("2")
    materials = _sub("3")
    institution = _sub("5")

    notes: List[Dict[str, str]] = []
    for note in public:
        notes.append({"classified_as": "Public note", "content": note})
    for note in nonpublic:
        notes.append({"classified_as": "Nonpublic note", "content": note})

    data: Dict[str, Any] = {}
    if action:
        data["action"] = action if len(action) > 1 else action[0]
    if action_id:
        data["actionId"] = action_id if len(action_id) > 1 else action_id[0]
    if date:
        data["date"] = date if len(date) > 1 else date[0]
    if interval:
        data["interval"] = interval if len(interval) > 1 else interval[0]
    if contingency:
        data["contingency"] = contingency if len(contingency) > 1 else contingency[0]
    if authorization:
        data["authorization"] = authorization if len(authorization) > 1 else authorization[0]
    if jurisdiction:
        data["jurisdiction"] = jurisdiction if len(jurisdiction) > 1 else jurisdiction[0]
    if method:
        data["method"] = method if len(method) > 1 else method[0]
    if site:
        data["site"] = site if len(site) > 1 else site[0]
    if agent:
        data["agent"] = agent if len(agent) > 1 else agent[0]
    if status:
        data["status"] = status if len(status) > 1 else status[0]
    if extent:
        data["extent"] = extent if len(extent) > 1 else extent[0]
    if unit_type:
        data["unitType"] = unit_type if len(unit_type) > 1 else unit_type[0]
    if uri:
        data["uri"] = uri if len(uri) > 1 else uri[0]
    if source:
        data["source"] = source if len(source) > 1 else source[0]
    if materials:
        data["materials"] = materials if len(materials) > 1 else materials[0]
    if institution:
        data["institution"] = institution if len(institution) > 1 else institution[0]
    if notes:
        data["notes"] = notes
    return data


def _format_conservation_label(action: Dict[str, Any]) -> str:
    if not action:
        return "Conservation action"
    action_val = action.get("action") or "Conservation action"
    if isinstance(action_val, list):
        action_text = "; ".join(str(v) for v in action_val if v)
    else:
        action_text = str(action_val)
    date_val = action.get("date")
    status_val = action.get("status")
    detail_parts = []
    if date_val:
        detail_parts.append(str(date_val))
    if status_val:
        detail_parts.append(str(status_val))
    if detail_parts:
        return f"{action_text} ({'; '.join(detail_parts)})"
    return action_text


def parse_marc_content(
    content: bytes,
    base_graph: Optional[GraphData] = None,
    base_id_to_label: Optional[Dict[str, str]] = None,
) -> Tuple[GraphData, Dict[str, str], List[str], Dict[str, int]]:
    stats = {
        "records": 0,
        "works_created": 0,
        "works_direct_matched": 0,
        "works_fuzzy_merged": 0,
        "works_fuzzy_linked": 0,
        "works_ambiguous": 0,
        "persons_created": 0,
        "persons_merged": 0,
        "persons_linked": 0,
        "orgs_created": 0,
        "orgs_merged": 0,
        "orgs_linked": 0,
        "actions_created": 0,
        "edges_created": 0,
    }
    errors: List[str] = []
    node_map: Dict[str, Node] = {}
    id_to_label: Dict[str, str] = {}

    if not pymarc_installed:
        errors.append("pymarc is not installed. Add it to requirements.txt to parse MARC21 .dat files.")
        return GraphData(nodes=[]), {}, errors, stats

    base_graph = base_graph or GraphData(nodes=[])
    base_id_to_label = base_id_to_label or {}
    index = _build_fuzzy_index(base_graph, base_id_to_label)

    person_cache: Dict[str, str] = {}
    org_cache: Dict[str, str] = {}

    def upsert_node(node: Node) -> None:
        existing = node_map.get(node.id)
        if existing is None:
            node_map[node.id] = node
            return
        if not existing.label and node.label:
            existing.label = node.label
        existing_types = list(existing.types or [])
        for node_type in node.types or []:
            if node_type not in existing_types:
                existing_types.append(node_type)
        existing.types = existing_types
        if isinstance(existing.metadata, dict) and isinstance(node.metadata, dict):
            for key, value in node.metadata.items():
                if key not in existing.metadata:
                    existing.metadata[key] = value
        elif not existing.metadata and node.metadata:
            existing.metadata = node.metadata
        existing_edge_keys = {(e.source, e.target, e.relationship) for e in existing.edges}
        for edge in node.edges:
            key = (edge.source, edge.target, edge.relationship)
            if key not in existing_edge_keys:
                existing.edges.append(edge)
                existing_edge_keys.add(key)

    def resolve_entity(
        label: str,
        node_type: str,
        merge_threshold: float,
        link_threshold: float,
        cache: Dict[str, str],
    ) -> str:
        normalized_label = _normalize_ris_text(label)
        if normalized_label in cache:
            return cache[normalized_label]

        candidates = index.get(node_type, [])
        if node_type != "Person":
            candidates = candidates + index.get("Unknown", [])
        match_id, score, ambiguous = _best_fuzzy_match(
            label,
            candidates,
            drop_stopwords=node_type != "Person",
            min_score=link_threshold,
        )
        if match_id and not ambiguous:
            if score >= merge_threshold:
                cache[normalized_label] = match_id
                if node_type == "Person":
                    stats["persons_merged"] += 1
                elif node_type == "Organization":
                    stats["orgs_merged"] += 1
                return match_id
            new_id = _make_marc_id(node_type.lower(), [normalized_label])
            same_as_edge = Edge(
                source=new_id,
                target=match_id,
                relationship="sameAs",
                weight=score,
            )
            node = Node(
                id=new_id,
                label=label,
                types=[node_type],
                metadata={"id": new_id, "prefLabel": {"en": label}, "type": [node_type]},
                edges=[same_as_edge],
            )
            upsert_node(node)
            id_to_label[new_id] = label
            cache[normalized_label] = new_id
            _add_candidate(index, new_id, label, node_type)
            if node_type == "Person":
                stats["persons_linked"] += 1
            elif node_type == "Organization":
                stats["orgs_linked"] += 1
            stats["edges_created"] += 1
            return new_id

        new_id = _make_marc_id(node_type.lower(), [normalized_label])
        node = Node(
            id=new_id,
            label=label,
            types=[node_type],
            metadata={"id": new_id, "prefLabel": {"en": label}, "type": [node_type]},
            edges=[],
        )
        upsert_node(node)
        id_to_label[new_id] = label
        cache[normalized_label] = new_id
        _add_candidate(index, new_id, label, node_type)
        if node_type == "Person":
            stats["persons_created"] += 1
        elif node_type == "Organization":
            stats["orgs_created"] += 1
        return new_id

    try:
        reader = MARCReader(io.BytesIO(content), to_unicode=True, force_utf8=True)
    except Exception as exc:
        errors.append(f"Unable to read MARC data: {exc}")
        return GraphData(nodes=[]), {}, errors, stats

    for record in reader:
        if record is None:
            errors.append("Encountered invalid MARC record.")
            continue
        stats["records"] += 1
        try:
            title_field = record.get_fields("245")
            title = ""
            if title_field:
                title_parts = _coerce_marc_list(title_field[0].get_subfields("a", "b", "n", "p"))
                title = " ".join(title_parts)
            if not title:
                title = record.title() or ""
            title = _clean_marc_text(title)

            control_number = ""
            control_prefix = ""
            if record["001"]:
                control_number = _clean_marc_text(record["001"].value())
            if record["003"]:
                control_prefix = _clean_marc_text(record["003"].value())

            publication_places: List[str] = []
            publishers: List[str] = []
            publication_dates: List[str] = []
            for field in record.get_fields("260", "264"):
                publication_places.extend(_coerce_marc_list(field.get_subfields("a")))
                publishers.extend(_coerce_marc_list(field.get_subfields("b")))
                publication_dates.extend(_coerce_marc_list(field.get_subfields("c")))

            year = _extract_year(publication_dates)
            authors_main = [
                _normalize_person_name(_format_marc_name(field))
                for field in record.get_fields("100")
                if _format_marc_name(field)
            ]
            authors_added = [
                _normalize_person_name(_format_marc_name(field))
                for field in record.get_fields("700")
                if _format_marc_name(field)
            ]

            orgs_main = [
                _format_marc_name(field)
                for field in record.get_fields("110", "111")
                if _format_marc_name(field)
            ]
            orgs_added = [
                _format_marc_name(field)
                for field in record.get_fields("710", "711")
                if _format_marc_name(field)
            ]

            work_id = ""
            if control_number:
                work_id = _make_marc_id("work", [control_prefix, control_number])
            else:
                work_id = _make_marc_id(
                    "work",
                    [title, year or "", ";".join(authors_main + authors_added)],
                )

            same_as_target: Optional[str] = None
            same_as_weight: Optional[float] = None
            if title:
                candidates = index.get("Work", []) + index.get("Unknown", [])
                match_id, score, ambiguous = _best_fuzzy_match(
                    title,
                    candidates,
                    drop_stopwords=True,
                    min_score=_WORK_LINK_THRESHOLD,
                )
                if match_id and not ambiguous:
                    if score >= _WORK_MERGE_THRESHOLD:
                        work_id = match_id
                        stats["works_fuzzy_merged"] += 1
                    else:
                        same_as_target = match_id
                        same_as_weight = score
                        stats["works_fuzzy_linked"] += 1
                elif ambiguous:
                    stats["works_ambiguous"] += 1

            isbn = []
            for field in record.get_fields("020"):
                isbn.extend(_coerce_marc_list(field.get_subfields("a")))
            isbn = [_strip_paren_suffix(val) for val in isbn if val]

            issn = []
            for field in record.get_fields("022"):
                issn.extend(_coerce_marc_list(field.get_subfields("a")))
            issn = [_strip_paren_suffix(val) for val in issn if val]

            other_ids = []
            for field in record.get_fields("024"):
                other_ids.extend(_coerce_marc_list(field.get_subfields("a")))

            languages: List[str] = []
            for field in record.get_fields("041"):
                languages.extend(_coerce_marc_list(field.get_subfields("a")))

            subjects: List[str] = []
            for field in record.get_fields("650", "651"):
                parts = _coerce_marc_list(field.get_subfields("a", "x", "y", "z"))
                if parts:
                    subjects.append(_clean_marc_text(" -- ".join(parts)))

            genres: List[str] = []
            for field in record.get_fields("655"):
                parts = _coerce_marc_list(field.get_subfields("a", "x", "y", "z"))
                if parts:
                    genres.append(_clean_marc_text(" -- ".join(parts)))

            abstract = ""
            for field in record.get_fields("520"):
                abstract = _clean_marc_text(" ".join(field.get_subfields("a")))
                if abstract:
                    break

            physical_desc = ""
            for field in record.get_fields("300"):
                physical_desc = _clean_marc_text(" ".join(field.get_subfields("a", "b", "c")))
                if physical_desc:
                    break

            action_fields = record.get_fields("583")
            actions: List[Dict[str, Any]] = []
            for field in action_fields:
                action = _parse_marc_583(field)
                if action:
                    actions.append(action)

            work_metadata: Dict[str, Any] = {
                "id": work_id,
                "prefLabel": {"en": title or work_id},
                "type": ["Work"],
                "source": "MARC21",
            }
            if title:
                work_metadata["title"] = title
            if control_number:
                work_metadata["controlNumber"] = control_number
            if control_prefix:
                work_metadata["controlNumberPrefix"] = control_prefix
            if year:
                work_metadata["year"] = year
            if publication_places:
                work_metadata["publicationPlace"] = publication_places
            if publishers:
                work_metadata["publisher"] = publishers
            if publication_dates:
                work_metadata["publicationDate"] = publication_dates
            if isbn:
                work_metadata["isbn"] = isbn
            if issn:
                work_metadata["issn"] = issn
            if other_ids:
                work_metadata["identifiers"] = other_ids
            if languages:
                work_metadata["language"] = languages
            if subjects:
                work_metadata["subjects"] = subjects
            if genres:
                work_metadata["genres"] = genres
            if abstract:
                work_metadata["abstract"] = abstract
            if physical_desc:
                work_metadata["physicalDescription"] = physical_desc
            notes = _collect_marc_notes(record)
            if notes:
                work_metadata["notes"] = notes
            if actions:
                work_metadata["583"] = actions

            work_node = Node(
                id=work_id,
                label=title or work_id,
                types=["Work"],
                metadata=work_metadata,
                edges=[],
            )
            if same_as_target and same_as_target != work_id:
                work_node.edges.append(
                    Edge(
                        source=work_id,
                        target=same_as_target,
                        relationship="sameAs",
                        weight=same_as_weight,
                    )
                )
                stats["edges_created"] += 1

            for author in authors_main:
                if not author:
                    continue
                person_id = resolve_entity(
                    author,
                    "Person",
                    _PERSON_MERGE_THRESHOLD,
                    _PERSON_LINK_THRESHOLD,
                    person_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=person_id, relationship="author"))
                stats["edges_created"] += 1

            for author in authors_added:
                if not author:
                    continue
                person_id = resolve_entity(
                    author,
                    "Person",
                    _PERSON_MERGE_THRESHOLD,
                    _PERSON_LINK_THRESHOLD,
                    person_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=person_id, relationship="contributor"))
                stats["edges_created"] += 1

            for org in orgs_main:
                if not org:
                    continue
                org_id = resolve_entity(
                    org,
                    "Organization",
                    _ORG_MERGE_THRESHOLD,
                    _ORG_LINK_THRESHOLD,
                    org_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=org_id, relationship="creator"))
                stats["edges_created"] += 1

            for org in orgs_added:
                if not org:
                    continue
                org_id = resolve_entity(
                    org,
                    "Organization",
                    _ORG_MERGE_THRESHOLD,
                    _ORG_LINK_THRESHOLD,
                    org_cache,
                )
                work_node.edges.append(Edge(source=work_id, target=org_id, relationship="contributor"))
                stats["edges_created"] += 1

            for action in actions:
                action_label = _format_conservation_label(action)
                action_seed = [work_id, action_label, str(action.get("date") or ""), str(action.get("agent") or "")]
                action_id = _make_marc_id("conservation", action_seed)
                action["id"] = action_id
                action_metadata = {
                    "id": action_id,
                    "prefLabel": {"en": action_label},
                    "type": ["ConservationAction"],
                    "source": "MARC21",
                }
                action_metadata.update(action)

                action_node = Node(
                    id=action_id,
                    label=action_label,
                    types=["ConservationAction"],
                    metadata=action_metadata,
                    edges=[],
                )

                agents = action.get("agent")
                agent_list = []
                if isinstance(agents, list):
                    agent_list = agents
                elif isinstance(agents, str):
                    agent_list = [agents]
                for agent in agent_list:
                    if not agent:
                        continue
                    person_id = resolve_entity(
                        _normalize_person_name(agent),
                        "Person",
                        _PERSON_MERGE_THRESHOLD,
                        _PERSON_LINK_THRESHOLD,
                        person_cache,
                    )
                    action_node.edges.append(
                        Edge(source=action_id, target=person_id, relationship="performedBy")
                    )
                    stats["edges_created"] += 1

                upsert_node(action_node)
                id_to_label[action_id] = action_label
                work_node.edges.append(
                    Edge(source=work_id, target=action_id, relationship="conservationAction")
                )
                stats["edges_created"] += 1
                stats["actions_created"] += 1

            upsert_node(work_node)
            id_to_label.setdefault(work_id, work_node.label)
            if work_id not in base_id_to_label:
                _add_candidate(index, work_id, work_node.label, "Work")
            stats["works_created"] += 1
        except Exception as exc:
            errors.append(f"MARC record {stats['records']}: {exc}")

    return GraphData(nodes=list(node_map.values())), id_to_label, errors, stats


def validate_with_shacl(
    rdf_graph: RDFGraph, shacl_data: str, shacl_format: str = "turtle"
) -> Tuple[bool, str]:
    if not pyshacl_installed:
        return False, "pySHACL is not installed."
    conforms, results_graph, report_text = validate(
        data_graph=rdf_graph,
        shacl_graph=RDFGraph().parse(data=shacl_data, format=shacl_format),
        inference="rdfs",
        debug=True,
    )
    return conforms, report_text


def dereference_uri(uri: str) -> Optional[Tuple[RDFGraph, int]]:
    try:
        headers = {
            "Accept": "application/ld+json, application/rdf+xml, text/turtle, application/n-triples;q=0.9"
        }
        response = requests.get(uri, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/ld+json" in content_type:
            fmt = "json-ld"
        elif "application/rdf+xml" in content_type:
            fmt = "xml"
        elif "text/turtle" in content_type or uri.endswith(".ttl"):
            fmt = "turtle"
        elif "application/n-triples" in content_type or uri.endswith(".nt"):
            fmt = "nt"
        else:
            fmt = "xml"
        new_graph = RDFGraph()
        new_graph.parse(data=response.text, format=fmt)
        triple_count = len(new_graph)
        logging.info(
            "Successfully dereferenced URI '%s' with %s triple(s) (format: %s).",
            uri,
            triple_count,
            fmt,
        )
        return new_graph, triple_count
    except Exception as exc:
        logging.error("Error dereferencing URI '%s': %s", uri, exc)
        return None


def apply_rdfs_reasoning(
    cg: Optional[ConjunctiveGraph],
    graph_data: Optional[GraphData] = None,
) -> Tuple[ConjunctiveGraph, int, Set[Tuple[Any, Any, Any]]]:
    if cg is None:
        cg = ConjunctiveGraph()
    elif not isinstance(cg, ConjunctiveGraph):
        wrapper = ConjunctiveGraph()
        wrapper += cg
        cg = wrapper

    if graph_data is not None:
        try:
            fresh = convert_graph_data_to_rdf(graph_data, cache_bust=time.time())
            asserted = cg.get_context(ASSERTED_CTX)
            asserted.remove((None, None, None))
            for triple in fresh.get_context(ASSERTED_CTX):
                asserted.add(triple)
        except Exception as exc:
            logging.error("Error syncing RDF graph with GraphData: %s", exc)

    inferred = cg.get_context(INFERRED_CTX)
    inferred.remove((None, None, None))
    seed_triples: Set[Tuple[Any, Any, Any]] = set()
    for ctx in cg.contexts():
        if ctx.identifier == INFERRED_CTX:
            continue
        for triple in ctx:
            seed_triples.add(triple)
    for triple in seed_triples:
        inferred.add(triple)

    initial_triples = set(inferred)
    if owlrl_installed:
        try:
            DeductiveClosure(RDFS_Semantics).expand(inferred)
        except Exception as exc:
            logging.error("Error during RDFS reasoning: %s", exc)
    else:
        logging.warning("owlrl not installed. Skipping RDFS reasoning.")
    new_triples = set(inferred) - initial_triples
    count = len(new_triples)
    logging.info("RDFS reasoning added %s new triple(s).", count)
    for triple in new_triples:
        logging.debug("New triple: %s", triple)
    return cg, count, new_triples


def compute_rdfs_signature(graph_data: GraphData, rdf_graph: Optional[ConjunctiveGraph]) -> Tuple[int, int, int, int]:
    node_count = len(graph_data.nodes)
    edge_count = 0
    edge_fingerprint = 0
    for node in graph_data.nodes:
        edge_count += len(node.edges)
        for edge in node.edges:
            edge_fingerprint ^= hash((edge.source, edge.target, edge.relationship))
    rdf_size = len(rdf_graph) if rdf_graph is not None else 0
    return node_count, edge_count, edge_fingerprint, rdf_size


def suggest_ontologies(rdf_graph: RDFGraph) -> List[str]:
    suggested = []
    known_vocabularies = {
        "http://xmlns.com/foaf/0.1/": "FOAF",
        "http://purl.org/dc/terms/": "Dublin Core",
        "http://schema.org/": "Schema.org",
        "http://www.w3.org/2004/02/skos/core#": "SKOS",
    }
    namespaces = dict(rdf_graph.namespaces())
    for ns_uri, name in known_vocabularies.items():
        if ns_uri in namespaces.values():
            suggested.append(name)
    return suggested


def load_schema_mapping(mapping_file: str) -> Dict[str, str]:
    try:
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        return mapping
    except Exception as exc:
        logging.error("Error loading schema mapping: %s", exc)
        return {}


# ------------------------------
# Data Processing Functions
# ------------------------------


def enhance_graph_data_from_triples(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    triples: Set[Tuple[Any, Any, Any]],
) -> Tuple[GraphData, Dict[str, str], Dict[str, int]]:
    node_map: Dict[str, Node] = {n.id: n for n in graph_data.nodes}
    edge_set: Set[Tuple[str, str, str]] = {
        (e.source, e.target, e.relationship) for n in graph_data.nodes for e in n.edges
    }
    stats = {"nodes": 0, "edges": 0, "types": 0, "inferred_types": 0, "labels": 0, "properties": 0}

    def _ensure_node(node_id: str) -> Node:
        if node_id in node_map:
            return node_map[node_id]
        label = id_to_label.get(node_id) or _local_name(node_id)
        new_node = Node(
            id=node_id,
            label=label,
            types=["Unknown"],
            metadata={"id": node_id, "prefLabel": {"en": label}, "type": ["Unknown"]},
            edges=[],
        )
        graph_data.nodes.append(new_node)
        node_map[node_id] = new_node
        id_to_label[node_id] = label
        stats["nodes"] += 1
        return new_node

    for s, p, o in triples:
        if isinstance(s, Literal):
            continue
        subj_id = str(s)
        subj = _ensure_node(subj_id)
        if not isinstance(subj.metadata, dict):
            subj.metadata = {}

        if p == RDF.type and isinstance(o, URIRef):
            raw_type = str(o)
            inferred_list = subj.metadata.get("inferredTypes")
            if inferred_list is None:
                subj.metadata["inferredTypes"] = []
                inferred_list = subj.metadata["inferredTypes"]
            elif not isinstance(inferred_list, list):
                subj.metadata["inferredTypes"] = [str(inferred_list)]
                inferred_list = subj.metadata["inferredTypes"]
            if raw_type not in inferred_list:
                inferred_list.append(raw_type)
                stats["inferred_types"] += 1

            canon = canonical_type(raw_type)
            if canon != "Unknown" and canon not in subj.types:
                subj.types.append(canon)
                subj.metadata.setdefault("type", [])
                if canon not in subj.metadata["type"]:
                    subj.metadata["type"].append(canon)
                stats["types"] += 1
            continue

        if isinstance(o, Literal):
            pred_local = _local_name(str(p)).lower()
        else:
            pred_local = ""

        if isinstance(o, Literal) and (p == RDFS.label or pred_local in {"label", "preflabel", "name"}):
            label = str(o)
            existing_label = subj.metadata.get("prefLabel", {}).get("en")
            if _label_is_unnamed(subj.label, subj.id):
                subj.label = label
            subj.metadata.setdefault("prefLabel", {})["en"] = label
            if subj.id not in id_to_label or _label_is_unnamed(id_to_label.get(subj.id), subj.id):
                id_to_label[subj.id] = subj.label if not _label_is_unnamed(subj.label, subj.id) else label
            if existing_label != label:
                stats["labels"] += 1
            continue

        pred_key = _local_name(str(p))
        if isinstance(o, URIRef):
            obj_id = str(o)
            _ensure_node(obj_id)
            edge_key = (subj_id, obj_id, pred_key)
            if edge_key not in edge_set:
                subj.edges.append(
                    Edge(source=subj_id, target=obj_id, relationship=pred_key, inferred=True)
                )
                edge_set.add(edge_key)
                stats["edges"] += 1
        elif isinstance(o, Literal):
            value = str(o)
            existing = subj.metadata.get(pred_key)
            if existing is None:
                subj.metadata[pred_key] = [value]
                stats["properties"] += 1
            elif isinstance(existing, list):
                if value not in existing:
                    existing.append(value)
                    stats["properties"] += 1
            else:
                if existing != value:
                    subj.metadata[pred_key] = [existing, value]
                    stats["properties"] += 1

    return graph_data, id_to_label, stats


def apply_render_cap(
    graph_data: GraphData,
    candidate_nodes: Set[str],
    max_nodes: int,
    selected_relationships: List[str],
    protected_nodes: Optional[List[str]] = None,
) -> Tuple[Set[str], Dict[str, int]]:
    stats = {"cap": max_nodes, "total": len(candidate_nodes), "trimmed": 0, "protected_trimmed": 0}
    if max_nodes <= 0 or len(candidate_nodes) <= max_nodes:
        return candidate_nodes, stats

    protected_nodes = protected_nodes or []
    seen: Set[str] = set()
    protected_unique: List[str] = []
    for node_id in protected_nodes:
        if node_id in candidate_nodes and node_id not in seen:
            protected_unique.append(node_id)
            seen.add(node_id)

    degree: Dict[str, int] = {node_id: 0 for node_id in candidate_nodes}
    rel_set = set(selected_relationships)
    for node in graph_data.nodes:
        if node.id not in candidate_nodes:
            continue
        for edge in node.edges:
            if edge.relationship not in rel_set:
                continue
            if edge.target in candidate_nodes:
                degree[node.id] = degree.get(node.id, 0) + 1
                degree[edge.target] = degree.get(edge.target, 0) + 1

    allowed: List[str] = list(protected_unique)
    if len(allowed) < max_nodes:
        remaining = candidate_nodes.difference(allowed)
        ranked = sorted(remaining, key=lambda nid: (-degree.get(nid, 0), nid))
        allowed.extend(ranked[: max_nodes - len(allowed)])

    allowed = allowed[:max_nodes]
    allowed_set = set(allowed)
    stats["trimmed"] = len(candidate_nodes) - len(allowed_set)
    if protected_unique:
        stats["protected_trimmed"] = len([n for n in protected_unique if n not in allowed_set])
    return allowed_set, stats


@st.cache_data(show_spinner=False)
@profile_time
def parse_entities_from_contents(file_contents: List[str]) -> Tuple[GraphData, Dict[str, str], List[str]]:
    nodes: List[Node] = []
    id_to_label: Dict[str, str] = {}
    errors: List[str] = []
    for idx, content in enumerate(file_contents):
        try:
            json_obj = json.loads(content)
            normalized = normalize_data(json_obj)
            subject_id: str = normalized["id"]
            label: str = normalized["prefLabel"]["en"]
            entity_types: List[str] = normalized.get("type", ["Unknown"])
            validation_errors = validate_entity(normalized)
            if validation_errors:
                errors.append(f"Entity '{subject_id}' errors: " + "; ".join(validation_errors))

            edges: List[Edge] = []
            for rel in CONFIG["RELATIONSHIP_CONFIG"]:
                values = normalized.get(rel, [])
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    weight = _extract_edge_weight(value)
                    normalized_id = normalize_relationship_value(rel, value)
                    if normalized_id:
                        edges.append(
                            Edge(
                                source=subject_id,
                                target=normalized_id,
                                relationship=rel,
                                weight=weight,
                            )
                        )
            linked_edges = _extract_linked_art_edges(json_obj, subject_id)
            if linked_edges:
                edge_keys = {(e.source, e.target, e.relationship) for e in edges}
                for edge in linked_edges:
                    key = (edge.source, edge.target, edge.relationship)
                    if key not in edge_keys:
                        edges.append(edge)
                        edge_keys.add(key)

            new_node = Node(
                id=subject_id,
                label=label,
                types=entity_types,
                metadata=normalized,
                edges=edges,
            )
            nodes.append(new_node)
            id_to_label[subject_id] = label
        except Exception as exc:
            err = f"File {idx}: {exc}\n{traceback.format_exc()}"
            errors.append(err)
            logging.error(err)
    return GraphData(nodes=nodes), id_to_label, errors


@st.cache_data(show_spinner=False)
@profile_time
def convert_graph_data_to_rdf(
    graph_data: GraphData, cache_bust: Optional[float] = None
) -> ConjunctiveGraph:
    cg = ConjunctiveGraph()
    cg.bind("ex", EX)
    asserted = cg.get_context(ASSERTED_CTX)
    for node in graph_data.nodes:
        subject = URIRef(node.id)
        label = node.metadata.get("prefLabel", {}).get("en", node.id)
        asserted.add((subject, RDFS.label, Literal(label, lang="en")))

        for t in node.types:
            asserted.add((subject, RDF.type, _type_uri(t)))

        for key, value in node.metadata.items():
            if key in ("id", "prefLabel", "type"):
                continue
            if key in CONFIG["RELATIONSHIP_CONFIG"]:
                if not isinstance(value, list):
                    value = [value]
                for v in value:
                    if isinstance(v, dict):
                        normalized_v = normalize_relationship_value(key, v)
                        if normalized_v and normalized_v.startswith("http"):
                            asserted.add((subject, EX[key], URIRef(normalized_v)))
                        else:
                            asserted.add((subject, EX[key], Literal(json.dumps(v))))
                    else:
                        if isinstance(v, str) and v.startswith("http"):
                            asserted.add((subject, EX[key], URIRef(v)))
                        else:
                            asserted.add((subject, EX[key], Literal(v)))
            else:
                if isinstance(value, str):
                    asserted.add((subject, EX[key], Literal(value)))
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict):
                            asserted.add((subject, EX[key], Literal(json.dumps(v))))
                        else:
                            asserted.add((subject, EX[key], Literal(v)))
                else:
                    if isinstance(value, dict):
                        asserted.add((subject, EX[key], Literal(json.dumps(value))))
                    else:
                        asserted.add((subject, EX[key], Literal(value)))

        for edge in node.edges:
            if getattr(edge, "inferred", False):
                continue
            asserted.add((subject, _predicate_uri(edge.relationship), URIRef(edge.target)))

    inferred = cg.get_context(INFERRED_CTX)
    for triple in asserted:
        inferred.add(triple)
    return cg


def run_sparql_query(query: str, rdf_graph: ConjunctiveGraph) -> Set[str]:
    result = rdf_graph.query(query, initNs={"rdf": RDF, "ex": EX})
    return {str(row[0]) for row in result if row[0] is not None}


def find_search_nodes(graph_data: GraphData, search_term: str) -> List[str]:
    term = (search_term or "").strip().lower()
    if not term or not graph_data.nodes:
        return []
    matches = []
    for node in graph_data.nodes:
        label = (node.label or "").lower()
        node_id = (node.id or "").lower()
        if term in label or term in node_id:
            matches.append(node.id)
    return matches


@st.cache_data(show_spinner=False)
@profile_time
def compute_centrality_measures(graph_data: GraphData) -> Dict[str, Dict[str, float]]:
    import networkx as nx

    G = nx.DiGraph()
    for node in graph_data.nodes:
        G.add_node(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            G.add_edge(edge.source, edge.target)

    n = max(1, G.number_of_nodes())
    degree = nx.degree_centrality(G)
    try:
        if n > 500:
            k = min(100, n)
            betweenness = nx.betweenness_centrality(G, k=k, seed=42)
        else:
            betweenness = nx.betweenness_centrality(G)
    except Exception as exc:
        logging.error("Betweenness centrality computation failed: %s", exc)
        betweenness = {node: 0.0 for node in G.nodes()}
    closeness = nx.closeness_centrality(G)

    try:
        if n > 0:
            UG = G.to_undirected()
            if UG.number_of_nodes() > 0:
                eigenvector = nx.eigenvector_centrality(UG, max_iter=1000)
            else:
                eigenvector = {node: 0.0 for node in G.nodes()}
        else:
            eigenvector = {}
    except Exception as exc:
        logging.error("Eigenvector centrality computation failed: %s", exc)
        eigenvector = {node: 0.0 for node in G.nodes()}

    pagerank = nx.pagerank(G)

    centrality = {}
    for node in G.nodes():
        centrality[node] = {
            "degree": degree.get(node, 0),
            "betweenness": betweenness.get(node, 0),
            "closeness": closeness.get(node, 0),
            "eigenvector": eigenvector.get(node, 0),
            "pagerank": pagerank.get(node, 0),
        }
    return centrality


def _safe_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    try:
        return float(np.percentile(values, percentile))
    except Exception:
        return 0.0


def compute_node_roles(
    graph_data: GraphData,
    hub_percentile: float = 85.0,
    bridge_percentile: float = 85.0,
    outlier_percentile: float = 15.0,
) -> Dict[str, List[str]]:
    """Classify nodes as hubs/bridges/outliers using centrality percentiles."""
    if not graph_data or not graph_data.nodes:
        return {}

    centrality = compute_centrality_measures(graph_data)
    if not centrality:
        return {}

    degree_vals = [float(metrics.get("degree", 0.0)) for metrics in centrality.values()]
    betweenness_vals = [float(metrics.get("betweenness", 0.0)) for metrics in centrality.values()]
    closeness_vals = [float(metrics.get("closeness", 0.0)) for metrics in centrality.values()]

    hub_threshold = _safe_percentile(degree_vals, hub_percentile)
    bridge_threshold = _safe_percentile(betweenness_vals, bridge_percentile)
    outlier_degree_threshold = _safe_percentile(degree_vals, outlier_percentile)
    outlier_closeness_threshold = _safe_percentile(closeness_vals, outlier_percentile)

    roles: Dict[str, List[str]] = {}
    for node_id, metrics in centrality.items():
        degree = float(metrics.get("degree", 0.0))
        betweenness = float(metrics.get("betweenness", 0.0))
        closeness = float(metrics.get("closeness", 0.0))
        node_roles: List[str] = []
        if degree >= hub_threshold and degree > 0:
            node_roles.append("hub")
        if betweenness >= bridge_threshold and betweenness > 0:
            node_roles.append("bridge")
        if degree <= outlier_degree_threshold and closeness <= outlier_closeness_threshold:
            node_roles.append("outlier")
        if node_roles:
            roles[node_id] = node_roles
    return roles


_ANOMALY_RESERVED_KEYS = {
    "id",
    "@id",
    "prefLabel",
    "label",
    "name",
    "type",
    "@type",
    "inferredTypes",
    "annotation",
    "annotation_html",
}


def _primary_node_type(node: Optional[Node]) -> str:
    if not node or not getattr(node, "types", None):
        return "Unknown"
    for node_type in node.types:
        canonical = canonical_type(node_type)
        if canonical != "Unknown":
            return canonical
    return canonical_type(node.types[0]) if node.types else "Unknown"


def _count_properties(metadata: Any) -> int:
    if not isinstance(metadata, dict):
        return 0
    count = 0
    for key, value in metadata.items():
        if key in _ANOMALY_RESERVED_KEYS:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        count += 1
    return count


def compute_anomaly_flags(
    graph_data: GraphData,
    rare_pattern_percentile: float = 10.0,
    low_property_percentile: float = 15.0,
) -> Dict[str, List[str]]:
    """Detect anomaly flags for nodes based on relationship patterns, types, and metadata coverage."""
    if not graph_data or not graph_data.nodes:
        return {}

    node_map = {node.id: node for node in graph_data.nodes if node.id}
    anomalies: Dict[str, Set[str]] = {}

    pattern_counts: Counter = Counter()
    for node in graph_data.nodes:
        if not node.id:
            continue
        source_type = _primary_node_type(node)
        for edge in node.edges:
            target_node = node_map.get(edge.target)
            target_type = _primary_node_type(target_node)
            pattern = (edge.relationship, source_type, target_type)
            pattern_counts[pattern] += 1

    if pattern_counts:
        counts = list(pattern_counts.values())
        if len(set(counts)) > 1:
            rare_threshold = _safe_percentile(counts, rare_pattern_percentile)
            rare_threshold = max(1, int(round(rare_threshold)))
            rare_patterns = {
                pattern for pattern, count in pattern_counts.items() if count <= rare_threshold
            }
            for node in graph_data.nodes:
                if not node.id:
                    continue
                source_type = _primary_node_type(node)
                for edge in node.edges:
                    target_node = node_map.get(edge.target)
                    target_type = _primary_node_type(target_node)
                    pattern = (edge.relationship, source_type, target_type)
                    if pattern in rare_patterns:
                        anomalies.setdefault(node.id, set()).add("rare_relationship_pattern")
                        if target_node:
                            anomalies.setdefault(target_node.id, set()).add(
                                "rare_relationship_pattern"
                            )

    for node in graph_data.nodes:
        if not node.id:
            continue
        canonical_types = {
            canonical_type(node_type)
            for node_type in (node.types or [])
            if canonical_type(node_type) != "Unknown"
        }
        if len(canonical_types) > 1:
            anomalies.setdefault(node.id, set()).add("conflicting_types")

    property_counts: Dict[str, int] = {}
    for node in graph_data.nodes:
        if not node.id:
            continue
        property_counts[node.id] = _count_properties(node.metadata)
    if property_counts:
        low_threshold = _safe_percentile(list(property_counts.values()), low_property_percentile)
        for node_id, count in property_counts.items():
            if count <= low_threshold:
                anomalies.setdefault(node_id, set()).add("low_property_coverage")

    return {node_id: sorted(list(flags)) for node_id, flags in anomalies.items() if flags}


def get_edge_relationship(source: str, target: str, graph_data: GraphData) -> List[str]:
    relationships = []
    for node in graph_data.nodes:
        if node.id == source:
            for edge in node.edges:
                if edge.target == target:
                    relationships.append(edge.relationship)
    return relationships


@st.cache_data(show_spinner=False)
@profile_time
def compute_probabilistic_graph_embeddings(
    graph_data: GraphData,
    dimensions=64,
    walk_length=30,
    num_walks=200,
    p=1,
    q=1,
    workers=4,
) -> Dict[str, List[float]]:
    import networkx as nx

    G = nx.DiGraph()
    for node in graph_data.nodes:
        G.add_node(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            G.add_edge(edge.source, edge.target)

    if G.number_of_nodes() == 0:
        logging.warning("Graph has no nodes. Skipping embeddings.")
        return {}

    if not node2vec_installed:
        logging.error("node2vec library not installed. Returning empty embeddings.")
        return {}

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[node].tolist()

    logging.info("Probabilistic Graph Embeddings computed for all nodes.")
    return embeddings


@st.cache_resource(show_spinner=False)
def get_st_model():
    if not sentence_transformer_installed:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        logging.error("Failed to load SentenceTransformer model: %s", exc)
        return None


@st.cache_data(show_spinner=False)
@profile_time
def compute_textual_semantic_embeddings(graph_data: GraphData) -> Dict[str, List[float]]:
    model = get_st_model()
    if model is None:
        logging.error("SentenceTransformer not available. Returning dummy embeddings.")
        return {node.id: [0.0] * 768 for node in graph_data.nodes}

    texts = []
    ids = []
    for node in graph_data.nodes:
        text = ""
        desc = node.metadata.get("description") if isinstance(node.metadata, dict) else None
        if isinstance(desc, dict):
            text = desc.get("en", "")
        elif isinstance(desc, str):
            text = desc
        if not text:
            text = node.label
        ids.append(node.id)
        texts.append(text)

    try:
        vectors = model.encode(texts, batch_size=64, show_progress_bar=False)
    except TypeError:
        vectors = model.encode(texts, batch_size=64)

    embeddings = {nid: vec.tolist() for nid, vec in zip(ids, vectors)}
    logging.info("Textual Semantic Embeddings computed for all nodes (batched).")
    return embeddings


def load_data_from_sparql(endpoint_url: str) -> Tuple[GraphData, Dict[str, str], List[str]]:
    errors = []
    nodes_dict = {}
    id_to_label = {}

    try:
        from rdflib.plugins.stores.sparqlstore import SPARQLStore

        store = SPARQLStore(endpoint_url)
        rdf_graph = RDFGraph(store=store)

        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
        results = rdf_graph.query(query)

        for row in results:
            s = str(row.s)
            p = str(row.p)
            o = str(row.o)
            if s not in nodes_dict:
                nodes_dict[s] = {"id": s, "prefLabel": {"en": s}, "type": ["Unknown"], "metadata": {}}

            if p in nodes_dict[s]["metadata"]:
                if isinstance(nodes_dict[s]["metadata"][p], list):
                    nodes_dict[s]["metadata"][p].append(o)
                else:
                    nodes_dict[s]["metadata"][p] = [nodes_dict[s]["metadata"][p], o]
            else:
                nodes_dict[s]["metadata"][p] = o

            if p.endswith("label"):
                nodes_dict[s]["prefLabel"] = {"en": o}

        nodes = []
        for s, data in nodes_dict.items():
            node = Node(
                id=data["id"],
                label=data["prefLabel"]["en"],
                types=data.get("type", ["Unknown"]),
                metadata=data["metadata"],
                edges=[],
            )
            nodes.append(node)
            id_to_label[s] = data["prefLabel"]["en"]

        graph_data = GraphData(nodes=nodes)

    except Exception as exc:
        errors.append(f"Error loading from SPARQL endpoint: {exc}")
        graph_data = GraphData(nodes=[])

    return graph_data, id_to_label, errors


# Note: Duplicate OCLC helpers removed; using the unified implementation defined earlier.


def graphdata_from_rdfgraph(rdf_graph: RDFGraph) -> GraphData:
    """Convert an rdflib Graph to the app's GraphData structure (best-effort)."""
    nodes: Dict[str, Node] = {}

    def _ensure_node(nid: str) -> Node:
        if nid not in nodes:
            nodes[nid] = Node(
                id=nid,
                label=nid,
                types=["Unknown"],
                metadata={"id": nid, "prefLabel": {"en": nid}, "type": ["Unknown"]},
                edges=[],
            )
        return nodes[nid]

    for s, p, o in rdf_graph:
        s_id = str(s)
        p_id = str(p)
        subj = _ensure_node(s_id)

        if p == RDFS.label and isinstance(o, Literal):
            subj.label = str(o)
            subj.metadata.setdefault("prefLabel", {}).update({"en": str(o)})
            continue
        if p == RDF.type and isinstance(o, URIRef):
            t = str(o).split("/")[-1] or "Unknown"
            subj.types = [t]
            subj.metadata["type"] = [t]
            continue

        if isinstance(o, URIRef):
            obj_id = str(o)
            _ensure_node(obj_id)
            rel_key = _local_name(p_id)
            subj.edges.append(Edge(source=s_id, target=obj_id, relationship=rel_key))
        else:
            key = _local_name(p_id)
            subj.metadata.setdefault(key, [])
            subj.metadata[key].append(str(o))

    return GraphData(nodes=list(nodes.values()))


def load_jsonld_into_session(jsonld_obj: Dict[str, Any]) -> Tuple[GraphData, Dict[str, str]]:
    """Parse JSON-LD into rdflib, convert to GraphData, and update session state."""
    rdf_g = RDFGraph().parse(data=json.dumps(jsonld_obj), format="json-ld")
    graph_data = graphdata_from_rdfgraph(rdf_g)
    id_to_label = {n.id: n.label for n in graph_data.nodes}
    st.session_state.graph_data = graph_data
    st.session_state.id_to_label = id_to_label
    try:
        st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
    except Exception as exc:
        st.error(f"Error converting loaded JSON-LD to RDF: {exc}")
    return graph_data, id_to_label
