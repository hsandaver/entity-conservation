"""Generic helpers (colors, formatting, profiling)."""

from __future__ import annotations

import functools
import html as html_lib
import json
import logging
import re
import time
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse, urlunparse

import numpy as np
from dateutil.parser import parse as parse_date
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS

from src.config import CONFIG, EX, NAMESPACE_PREFIXES, RDFS_PREDICATE_SUFFIXES, TYPE_SYNONYMS
from src.models import Edge, GraphData, Node

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _blend_hex(color: str, target: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    r1, g1, b1 = _hex_to_rgb(color)
    r2, g2, b2 = _hex_to_rgb(target)
    r = round(r1 + (r2 - r1) * ratio)
    g = round(g1 + (g2 - g1) * ratio)
    b = round(b1 + (b2 - b1) * ratio)
    return _rgb_to_hex((r, g, b))


def _make_node_color(base: str) -> Dict[str, Any]:
    return {
        "background": base,
        "border": _blend_hex(base, "#1F2A37", 0.35),
        "highlight": {
            "background": _blend_hex(base, "#FFFFFF", 0.18),
            "border": _blend_hex(base, "#0F172A", 0.45),
        },
        "hover": {
            "background": _blend_hex(base, "#FFFFFF", 0.12),
            "border": _blend_hex(base, "#0F172A", 0.4),
        },
    }


def _relative_luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb(hex_color)

    def _channel(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r_l, g_l, b_l = _channel(r), _channel(g), _channel(b)
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def _contrast_ratio(lum_a: float, lum_b: float) -> float:
    lighter, darker = (lum_a, lum_b) if lum_a >= lum_b else (lum_b, lum_a)
    return (lighter + 0.05) / (darker + 0.05)


def _pick_label_color(bg_hex: str, dark: str = "#0B0B0B", light: str = "#F8F6F1") -> str:
    try:
        lum_bg = _relative_luminance(bg_hex)
        lum_dark = _relative_luminance(dark)
        lum_light = _relative_luminance(light)
    except ValueError:
        return dark
    contrast_dark = _contrast_ratio(lum_bg, lum_dark)
    contrast_light = _contrast_ratio(lum_bg, lum_light)
    return dark if contrast_dark >= contrast_light else light


def _make_edge_color(base: str) -> Dict[str, Any]:
    return {
        "color": base,
        "highlight": _blend_hex(base, "#FFFFFF", 0.15),
        "hover": _blend_hex(base, "#FFFFFF", 0.08),
        "opacity": 0.78,
    }


def _label_is_unnamed(label: Any, node_id: Any) -> bool:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return True
    label_str = str(label).strip()
    if not label_str:
        return True
    if label_str.lower() in {"unknown", "unnamed", "untitled"}:
        return True
    node_id_str = "" if node_id is None else str(node_id).strip()
    if label_str == node_id_str:
        return True
    lower_label = label_str.lower()
    if re.match(r"^[a-z]+://", lower_label) or lower_label.startswith("urn:"):
        return True
    if node_id_str and label_str == _local_name(node_id_str):
        return True
    return False


def refresh_label_index(
    graph_data: Optional["GraphData"],
    id_to_label: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    label_map = dict(id_to_label or {})
    if not graph_data or not getattr(graph_data, "nodes", None):
        return label_map
    for node in graph_data.nodes:
        node_id = getattr(node, "id", None)
        if not node_id:
            continue
        node_id = str(node_id)
        label = getattr(node, "label", None)
        meta_label = None
        metadata = getattr(node, "metadata", None)
        if isinstance(metadata, dict):
            meta_label = _extract_pref_label(metadata)
        candidate = meta_label or label
        if not candidate or _label_is_unnamed(candidate, node_id):
            continue
        if _label_is_unnamed(label, node_id):
            node.label = candidate
        existing = label_map.get(node_id)
        if not existing or _label_is_unnamed(existing, node_id):
            label_map[node_id] = candidate
    return label_map


def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info("[PROFILE] Function '%s' executed in %.3f seconds", func.__name__, elapsed)
        return result

    return wrapper


def remove_fragment(uri: str) -> str:
    try:
        parsed = urlparse(uri)
        return urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
        )
    except Exception as exc:  # pragma: no cover - defensive log
        logging.error("Error removing fragment from %s: %s", uri, exc)
        return uri


def _normalize_getty_ulan(uri: str) -> str:
    if not isinstance(uri, str):
        return uri
    text = uri.strip()
    if not text:
        return uri
    text = text.replace("https://vocab.getty.edu/page/ulan/", "http://vocab.getty.edu/ulan/")
    text = text.replace("http://vocab.getty.edu/page/ulan/", "http://vocab.getty.edu/ulan/")
    text = text.replace("https://vocab.getty.edu/ulan/", "http://vocab.getty.edu/ulan/")
    return text.rstrip("/")


def normalize_relationship_value(rel: str, value: Any) -> Optional[str]:
    if isinstance(value, dict):
        if rel in {
            "spouse",
            "studentOf",
            "employedBy",
            "educatedAt",
            "contributor",
            "draftsman",
            "creator",
            "author",
            "owner",
            "dedicatee",
            "illustrator",
            "artist",
            "editor",
            "designer",
            "composer",
            "translator",
        }:
            return remove_fragment(value.get("carriedOutBy", value.get("id", "")))
        if rel == "succeededBy":
            return remove_fragment(value.get("resultedIn", ""))
        if rel == "precededBy":
            return remove_fragment(value.get("resultedFrom", ""))
        if rel == "foundedBy":
            return remove_fragment(value.get("carriedOutBy", value.get("founder", value.get("id", ""))))
        normalized = remove_fragment(value.get("id", ""))
        return _normalize_getty_ulan(normalized) if rel == "sameAs" else normalized
    if isinstance(value, str):
        normalized = remove_fragment(value)
        return _normalize_getty_ulan(normalized) if rel == "sameAs" else normalized
    return None


def _extract_pref_label(data: Dict[str, Any]) -> Optional[str]:
    pref = data.get("prefLabel")
    if isinstance(pref, dict):
        en_val = pref.get("en")
        if isinstance(en_val, str) and en_val.strip():
            return en_val.strip()
        for val in pref.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    elif isinstance(pref, str) and pref.strip():
        return pref.strip()
    elif isinstance(pref, list):
        for item in pref:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for key in ("@value", "value", "_label", "label", "content"):
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        return raw.strip()

    for key in ("_label", "label", "name"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    identified = data.get("identified_by")
    if isinstance(identified, dict):
        identified = [identified]
    if isinstance(identified, list):
        preferred: List[str] = []
        fallback: List[str] = []
        for entry in identified:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content") or entry.get("label") or entry.get("_label")
            if not isinstance(content, str) or not content.strip():
                continue
            is_preferred = False
            classified = entry.get("classified_as")
            if isinstance(classified, dict):
                classified = [classified]
            if isinstance(classified, list):
                for c in classified:
                    if isinstance(c, str):
                        if "300404670" in c:
                            is_preferred = True
                            break
                        continue
                    if not isinstance(c, dict):
                        continue
                    label = str(c.get("_label") or c.get("label") or "").lower()
                    cid = str(c.get("id") or "")
                    if "preferred" in label or cid.endswith("300404670"):
                        is_preferred = True
                        break
            if is_preferred:
                preferred.append(content.strip())
            else:
                fallback.append(content.strip())
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
    return None


def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if ("id" not in data or not data["id"]) and "@id" in data:
        data["id"] = data.get("@id")

    # Some inputs are not individual entities but wrapper payloads (e.g. saved graph exports).
    # Instead of crashing the pipeline, wrap them into a synthetic entity.
    if "id" not in data or not data["id"]:
        graph_obj = data.get("graph") if isinstance(data, dict) else None
        if isinstance(graph_obj, dict) and isinstance(graph_obj.get("nodes"), list):
            saved_at = data.get("saved_at") or data.get("savedAt")
            synthetic_id = f"urn:entity-explorer:graph-export:{saved_at or int(time.time())}"
            logging.warning(
                "normalize_data received a graph export wrapper without an id; using synthetic id %s",
                synthetic_id,
            )
            return {
                "id": synthetic_id,
                "prefLabel": {"en": "Saved Graph"},
                "type": ["GraphExport"],
                "version": data.get("version"),
                "saved_at": saved_at,
                "graph": graph_obj,
            }

        # Provide more diagnostic info about the problematic data
        data_preview = str(data)[:400] if data else "empty dict"
        raise ValueError(
            "Entity is missing an 'id'. If this is a wrapper payload, pass an entity object instead. "
            f"Data: {data_preview}"
        )

    data["id"] = remove_fragment(data.get("id", ""))
    label_candidate = _extract_pref_label(data)
    if isinstance(data.get("prefLabel"), dict):
        pref = data["prefLabel"]
        if not pref.get("en", "").strip():
            pref["en"] = label_candidate or _local_name(data["id"]) or data.get("id", "unknown")
        data["prefLabel"] = pref
    else:
        data["prefLabel"] = {"en": label_candidate or _local_name(data["id"]) or data.get("id", "unknown")}
    if "type" not in data and "@type" in data:
        data["type"] = data.get("@type")
    if "type" in data:
        data["type"] = data["type"] if isinstance(data["type"], list) else [data["type"]]
    else:
        data["type"] = ["Unknown"]

    for time_field in ["dateOfBirth", "dateOfDeath"]:
        if time_field in data:
            try:
                if isinstance(data[time_field], list):
                    data[time_field] = [
                        {"time:inXSDDateTimeStamp": {"@value": parse_date(item).isoformat()}}
                        if isinstance(item, str)
                        else item
                        for item in data[time_field]
                    ]
                elif isinstance(data[time_field], str):
                    data[time_field] = [
                        {"time:inXSDDateTimeStamp": {"@value": parse_date(data[time_field]).isoformat()}}
                    ]
            except Exception as exc:  # pragma: no cover - best effort
                logging.error("Error parsing %s for %s: %s", time_field, data["id"], exc)

    for rel in list(data.keys()):
        if rel not in CONFIG["RELATIONSHIP_CONFIG"]:
            continue
        if rel in ["educatedAt", "employedBy", "dateOfBirth", "dateOfDeath"]:
            continue
        values = data[rel]
        normalized_values = []
        if not isinstance(values, list):
            values = [values]
        for value in values:
            normalized_id = normalize_relationship_value(rel, value)
            if normalized_id:
                normalized_values.append(normalized_id)
                logging.debug("Normalized relationship '%s': %s -> %s", rel, data["id"], normalized_id)
        data[rel] = normalized_values
    return data


def is_valid_iiif_manifest(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
        if parsed.path.endswith("manifest") or parsed.path.endswith("manifest.json"):
            return True
        query_params = parse_qs(parsed.query)
        if "manifest" in query_params:
            return True
        return False
    except Exception as exc:  # pragma: no cover - defensive log
        logging.error("Error validating IIIF manifest URL '%s': %s", url, exc)
        return False


def validate_entity(entity: Dict[str, Any]) -> List[str]:
    errors = []
    if "id" not in entity or not entity["id"].strip():
        errors.append("Missing 'id'.")
    if "prefLabel" not in entity or not entity["prefLabel"].get("en", "").strip():
        errors.append("Missing 'prefLabel' with English label.")
    return errors


def format_metadata(metadata: Dict[str, Any]) -> str:
    formatted = ""
    for key, value in metadata.items():
        if key in ("prefLabel", "inferredTypes"):
            continue
        formatted += f"- **{key}**: "
        if isinstance(value, list):
            formatted += "\n" + "\n".join([f"  - {v}" for v in value])
        elif isinstance(value, str):
            if value.startswith("http"):
                formatted += f"[{value}]({value})"
            else:
                formatted += f"{value}"
        elif isinstance(value, dict):
            formatted += "\n" + "\n".join(
                [f"  - **{subkey}**: {subvalue}" for subkey, subvalue in value.items()]
            )
        else:
            formatted += str(value)
        formatted += "\n"
    return formatted


def _truncate_text(text: str, max_len: int = 160) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _summarize_value(value: Any, max_items: int = 6, max_len: int = 160) -> str:
    if isinstance(value, list):
        items = [str(v) for v in value]
        suffix = ""
        if len(items) > max_items:
            suffix = f" ... (+{len(items) - max_items} more)"
            items = items[:max_items]
        return _truncate_text(", ".join(items) + suffix, max_len)
    if isinstance(value, dict):
        try:
            rendered = json.dumps(value, ensure_ascii=True)
        except Exception:
            rendered = str(value)
        return _truncate_text(rendered, max_len)
    return _truncate_text(value, max_len)


_META_KEY_LABELS: Dict[str, str] = {
    "@context": "Context",
    "_label": "Label",
    "born": "Born",
    "died": "Died",
    "identified_by": "Names",
    "classified_as": "Roles",
    "referred_to_by": "Notes",
    "subject_of": "Notes",
    "sameAs": "Same As",
    "rdfs:seeAlso": "See Also",
    "skos:inScheme": "In Scheme",
    "la:related_from_by": "Relationships",
    "la:related_to_by": "Relationships",
    "la:relates_to": "Related To",
    "timespan": "Date",
    "took_place_at": "Place",
    "language": "Language",
    "583": "Conservation Actions (MARC 583)",
}

_META_KEY_PRIORITY: Dict[str, int] = {
    "_label": 0,
    "identified_by": 1,
    "classified_as": 2,
    "583": 3,
    "born": 3,
    "died": 4,
    "referred_to_by": 5,
    "subject_of": 6,
    "la:related_from_by": 7,
    "sameAs": 8,
    "rdfs:seeAlso": 9,
    "skos:inScheme": 10,
    "@context": 11,
}


def _humanize_meta_key(key: Any) -> str:
    key_str = str(key)
    if key_str in _META_KEY_LABELS:
        return _META_KEY_LABELS[key_str]
    cleaned = key_str.strip()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    cleaned = cleaned.replace("_", " ").replace(":", " ")
    cleaned = re.sub(r"(?<!^)([A-Z])", r" \1", cleaned)
    return " ".join(word.capitalize() for word in cleaned.split())


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_uri(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(("http://", "https://"))


def _format_link(iri: str, label: Optional[str] = None) -> str:
    if not isinstance(iri, str) or not iri.strip():
        return ""
    safe_iri = html_lib.escape(iri)
    link_label = label or _shorten_iri(iri, max_len=52)
    return (
        f"<a class='meta-link' href='{safe_iri}' target='_blank' rel='noopener noreferrer'>"
        f"{html_lib.escape(link_label)}</a>"
    )


def _chip_html(text: str, href: Optional[str] = None, primary: bool = False) -> str:
    classes = ["meta-chip"]
    if primary:
        classes.append("meta-chip-primary")
    class_attr = " ".join(classes)
    label = html_lib.escape(text)
    if href and _is_uri(href):
        return (
            f"<a class='{class_attr}' href='{html_lib.escape(href)}' "
            f"target='_blank' rel='noopener noreferrer'>{label}</a>"
        )
    return f"<span class='{class_attr}'>{label}</span>"


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _resolve_label_from_lookup(value: Any, label_lookup: Optional[Dict[str, str]]) -> Optional[str]:
    if not label_lookup:
        return None
    def _candidates(text: str) -> List[str]:
        if not text:
            return []
        pool: List[str] = [text]
        if ":" in text and not text.startswith(("http://", "https://")):
            prefix, local = text.split(":", 1)
            for ns, pref in NAMESPACE_PREFIXES.items():
                if pref == prefix:
                    pool.append(f"{ns}{local}")
                    break
        for item in list(pool):
            cleaned = remove_fragment(item)
            if cleaned != item:
                pool.append(cleaned)
        for item in list(pool):
            if item.startswith("https://"):
                pool.append("http://" + item[len("https://") :])
            elif item.startswith("http://"):
                pool.append("https://" + item[len("http://") :])
        for item in list(pool):
            if item.endswith("/"):
                pool.append(item.rstrip("/"))
            else:
                pool.append(item + "/")
        for item in list(pool):
            normalized = _normalize_getty_ulan(item)
            if normalized and normalized != item:
                pool.append(normalized)
            for vocab in ("ulan", "aat", "tgn", "language"):
                base = f"vocab.getty.edu/{vocab}/"
                page = f"vocab.getty.edu/page/{vocab}/"
                if base in item and page not in item:
                    pool.append(item.replace(base, page))
                if page in item and base not in item:
                    pool.append(item.replace(page, base))
        for item in list(pool):
            for ns, pref in NAMESPACE_PREFIXES.items():
                if item.startswith(ns):
                    pool.append(f"{pref}:{item[len(ns):]}")
        seen: Set[str] = set()
        deduped: List[str] = []
        for item in pool:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    if isinstance(value, str):
        for candidate in _candidates(value):
            label = label_lookup.get(candidate)
            if label:
                return label
        return None
    if isinstance(value, dict):
        ident = value.get("id") or value.get("@id")
        if isinstance(ident, str):
            for candidate in _candidates(ident):
                label = label_lookup.get(candidate)
                if label:
                    return label
    return None


def _make_modal_id(key: Optional[str], suffix: str) -> str:
    base = f"{key or 'meta'}-{suffix}"
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", base.lower()).strip("-")
    return f"meta-modal-{safe}-{next(_META_MODAL_COUNTER)}"


def _build_modal(trigger_label: str, title: str, body_html: str, modal_id: str) -> str:
    return (
        f"<a class='meta-more' href='#{modal_id}'>{html_lib.escape(trigger_label)}</a>"
        f"<div id='{modal_id}' class='meta-modal'>"
        "<div class='meta-modal-card'>"
        "<div class='meta-modal-header'>"
        f"<div class='meta-modal-title'>{html_lib.escape(title)}</div>"
        "<a class='meta-modal-close' href='#'>Close</a>"
        "</div>"
        f"<div class='meta-modal-body'>{body_html}</div>"
        "</div>"
        "</div>"
    )


def _build_modal_chips(items: List[Any], max_len: int = 70) -> str:
    chips = []
    for item in items:
        if isinstance(item, dict):
            ident = item.get("id") or item.get("@id")
            label = _extract_label_text(item, max_len=max_len) or (
                _shorten_iri(ident, max_len=max_len) if isinstance(ident, str) else None
            )
            href = ident if _is_uri(ident) else None
        else:
            if isinstance(item, str) and _is_uri(item):
                label = _shorten_iri(item, max_len=max_len)
            else:
                label = _extract_label_text(item, max_len=max_len)
            href = item if isinstance(item, str) and _is_uri(item) else None
        if label:
            chips.append(_chip_html(label, href=href))
    if not chips:
        return "<span class='meta-muted'>None</span>"
    return f"<div class='meta-modal-list'>{''.join(chips)}</div>"


def _build_modal_target_chips(targets: List[Tuple[str, Optional[str]]]) -> str:
    chips = [_chip_html(label, href=href) for label, href in targets if label]
    if not chips:
        return "<span class='meta-muted'>None</span>"
    return f"<div class='meta-modal-list'>{''.join(chips)}</div>"


def _extract_lang_text(
    value: Any,
    preferred_langs: Tuple[str, ...] = ("en", "la", "fr", "it", "de", "es"),
) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    if isinstance(value, dict):
        for lang in preferred_langs:
            if lang in value:
                lang_val = value.get(lang)
                if isinstance(lang_val, list) and lang_val:
                    lang_val = lang_val[0]
                if isinstance(lang_val, str) and lang_val.strip():
                    return lang_val.strip()
        for lang_val in value.values():
            if isinstance(lang_val, list) and lang_val:
                lang_val = lang_val[0]
            if isinstance(lang_val, str) and lang_val.strip():
                return lang_val.strip()
    return None


def _format_lang_text(value: Any, max_len: int = 220) -> str:
    text = _extract_lang_text(value)
    if not text:
        return "<span class='meta-muted'>None</span>"
    return html_lib.escape(_truncate_text(text, max_len))


def _extract_time_label(value: Any) -> Optional[str]:
    if isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, dict):
        return None
    date_value = value.get("time:inXSDDateTimeStamp")
    if isinstance(date_value, dict):
        raw = date_value.get("@value")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = parse_date(raw).date()
                if parsed.month == 1 and parsed.day == 1:
                    return str(parsed.year)
                return parsed.isoformat()
            except Exception:
                return raw.strip()
    date_desc = value.get("time:inDateTime")
    if isinstance(date_desc, dict):
        year = date_desc.get("time:year")
        if isinstance(year, dict):
            year_val = year.get("@value")
            if isinstance(year_val, str) and year_val.strip():
                return year_val.strip()
        if isinstance(year, str) and year.strip():
            return year.strip()
    return None


def _format_time_list(value: Any, max_items: int = 6, modal_key: Optional[str] = None) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    labels = []
    for item in items:
        label = _extract_time_label(item)
        if label:
            labels.append(label)
    labels = _dedupe_preserve(labels)
    if not labels:
        return _format_meta_value_html(value)
    chips = [_chip_html(label) for label in labels[:max_items]]
    if len(labels) > max_items:
        modal_id = _make_modal_id(modal_key or "time", "time")
        modal_body = _build_modal_chips(labels, max_len=32)
        chips.append(
            _build_modal(
                f"+{len(labels) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'Dates')} ({len(labels)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-chips'>{''.join(chips)}</div>"


def _format_role_item(
    item: Any,
    label_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(item, dict):
        actor = item.get("carriedOutBy") or item.get("carried_out_by") or item.get("carriedOutBy")
        label = _resolve_label_from_lookup(actor, label_lookup)
        if not label:
            if isinstance(actor, str) and _is_uri(actor):
                label = _shorten_iri(actor, max_len=60)
            elif isinstance(actor, dict):
                label = _extract_label_text(actor, max_len=60)
        date = _extract_time_label(item.get("startDate")) or _extract_time_label(item.get("endDate"))
        if label and date:
            label = f"{label} - {date}"
        href = actor if isinstance(actor, str) and _is_uri(actor) else None
        if not label:
            label = _summarize_value(item, max_items=3, max_len=60)
        return label, href
    if isinstance(item, str):
        label = _resolve_label_from_lookup(item, label_lookup) or (
            _shorten_iri(item, max_len=60) if _is_uri(item) else item
        )
        href = item if _is_uri(item) else None
        return label, href
    return _summarize_value(item, max_items=3, max_len=60), None


def _format_role_list(
    value: Any,
    max_items: int = 6,
    modal_key: Optional[str] = None,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    chips = []
    for item in items[:max_items]:
        label, href = _format_role_item(item, label_lookup=label_lookup)
        if label:
            chips.append(_chip_html(label, href=href))
    if len(items) > max_items:
        modal_id = _make_modal_id(modal_key or "roles", "roles")
        modal_targets = []
        for item in items:
            label, href = _format_role_item(item, label_lookup=label_lookup)
            if label:
                modal_targets.append((label, href))
        modal_body = _build_modal_target_chips(modal_targets)
        chips.append(
            _build_modal(
                f"+{len(items) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'Roles')} ({len(items)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-chips'>{''.join(chips)}</div>"


def _extract_label_text(value: Any, max_len: int = 90) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (Literal, URIRef)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    if isinstance(value, dict):
        if _extract_date_text(value):
            date_text = _extract_date_text(value)
            if date_text:
                return _truncate_text(date_text, max_len)
        for time_key in ("time:inDateTime", "inDateTime", "time:hasBeginning", "time:hasEnd"):
            if time_key in value:
                nested = _extract_label_text(value.get(time_key), max_len=max_len)
                if nested:
                    return _truncate_text(nested, max_len)
        label = _extract_pref_label(value)
        if isinstance(label, str) and label.strip():
            return _truncate_text(label.strip(), max_len)
        for key in ("_label", "label", "content", "name", "title", "@value", "value", "rdf:value", "rdf:label"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return _truncate_text(raw.strip(), max_len)
        ident = value.get("id") or value.get("@id")
        if isinstance(ident, str) and ident.strip():
            return _shorten_iri(ident.strip(), max_len=max_len)
    return None


def _extract_date_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                return cleaned
            return _extract_date_text(parsed) or cleaned
        return cleaned
    if isinstance(value, list):
        for item in value:
            text = _extract_date_text(item)
            if text:
                return text
        return None
    if isinstance(value, dict):
        for key in ("time:inXSDDateTimeStamp", "inXSDDateTimeStamp"):
            if key in value:
                text = _extract_date_text(value.get(key))
                if text:
                    if "T" in text:
                        return text.split("T", 1)[0]
                    return text
        for key in (
            "time:inDateTime",
            "inDateTime",
            "time:hasBeginning",
            "time:hasEnd",
            "time:hasBeginning",
            "time:hasEnd",
        ):
            if key in value:
                text = _extract_date_text(value.get(key))
                if text:
                    return text
        for key in ("@value", "value", "rdf:value"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            if raw is not None:
                nested = _extract_date_text(raw)
                if nested:
                    return nested
        if "type" in value and isinstance(value.get("type"), list):
            type_values = [str(t).lower() for t in value.get("type") if t]
            if any("instant" in t or "date" in t or "time" in t for t in type_values):
                for key in ("inDateTime", "time:inDateTime"):
                    if key in value:
                        text = _extract_date_text(value.get(key))
                        if text:
                            return text
                for key, subvalue in value.items():
                    if key == "type":
                        continue
                    text = _extract_date_text(subvalue)
                    if text:
                        return text
        # Handle time:DateTimeDescription style objects
        if any(k.startswith("time:") for k in value.keys()):
            year = value.get("time:year") or value.get("year")
            month = value.get("time:month") or value.get("month")
            day = value.get("time:day") or value.get("day")

            def _get_val(raw: Any) -> Optional[str]:
                if isinstance(raw, dict):
                    raw_val = raw.get("@value") or raw.get("value")
                    if isinstance(raw_val, str) and raw_val.strip():
                        return raw_val.strip()
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
                return None

            def _clean_month(text: Optional[str]) -> Optional[str]:
                if not text:
                    return None
                cleaned = text.strip()
                cleaned = cleaned.replace("--", "")
                return cleaned.zfill(2) if cleaned.isdigit() else None

            def _clean_day(text: Optional[str]) -> Optional[str]:
                if not text:
                    return None
                cleaned = text.strip()
                cleaned = cleaned.replace("----", "").replace("--", "")
                return cleaned.zfill(2) if cleaned.isdigit() else None

            year_text = _get_val(year)
            month_text = _clean_month(_get_val(month))
            day_text = _clean_day(_get_val(day))
            if year_text and month_text and day_text:
                return f"{year_text}-{month_text}-{day_text}"
            if year_text and month_text:
                return f"{year_text}-{month_text}"
            if year_text:
                return year_text
        begin = value.get("begin_of_the_begin") or value.get("begin")
        end = value.get("end_of_the_end") or value.get("end")
        date_text = _format_date_range(begin, end)
        if date_text:
            return date_text
    return None


def _extract_first_label(values: Any, fallback: Optional[str] = None) -> Optional[str]:
    for item in _coerce_list(values):
        label = _extract_label_text(item)
        if label:
            return label
    return fallback


def _format_date_range(begin: Optional[str], end: Optional[str]) -> Optional[str]:
    if not begin and not end:
        return None
    try:
        begin_dt = parse_date(begin).date() if begin else None
        end_dt = parse_date(end).date() if end else None
    except Exception:
        begin_dt = None
        end_dt = None
    if begin_dt and end_dt:
        if (
            begin_dt.year == end_dt.year
            and begin_dt.month == 1
            and begin_dt.day == 1
            and end_dt.month == 12
            and end_dt.day == 31
        ):
            return str(begin_dt.year)
        if begin_dt == end_dt:
            return begin_dt.isoformat()
        return f"{begin_dt.isoformat()} - {end_dt.isoformat()}"
    if begin_dt:
        return begin_dt.isoformat()
    if end_dt:
        return end_dt.isoformat()
    return None


def _format_timespan_details(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    begin = value.get("begin_of_the_begin") or value.get("begin")
    end = value.get("end_of_the_end") or value.get("end")
    return _format_date_range(begin, end)


def _format_event_details(value: Any) -> str:
    if not isinstance(value, dict):
        return _format_meta_value_html(value)
    timespan = value.get("timespan")
    if isinstance(timespan, list):
        timespan = timespan[0] if timespan else None
    date_text = None
    if isinstance(timespan, dict):
        date_text = _format_timespan_details(timespan)
    place_labels = []
    for place in _coerce_list(value.get("took_place_at")):
        label = _extract_label_text(place, max_len=60)
        if label:
            place_labels.append(label)
    rows = []
    if not date_text:
        date_text = _extract_date_text(value)
    if date_text:
        rows.append(
            "<div class='meta-kv'><span>Date</span>"
            f"<strong>{html_lib.escape(date_text)}</strong></div>"
        )
    if place_labels:
        place_text = _truncate_text(", ".join(place_labels), 80)
        rows.append(
            "<div class='meta-kv'><span>Place</span>"
            f"<strong>{html_lib.escape(place_text)}</strong></div>"
        )
    if not rows:
        return "<span class='meta-muted'>No details</span>"
    return f"<div class='meta-kv-list'>{''.join(rows)}</div>"


def _format_chip_section(
    title: str,
    items: List[str],
    max_items: int,
    primary: bool = False,
    modal_key: Optional[str] = None,
) -> str:
    if not items:
        return ""
    items = _dedupe_preserve(items)
    chips = [_chip_html(item, primary=primary) for item in items[:max_items]]
    if len(items) > max_items:
        modal_id = _make_modal_id(modal_key or title, "chips")
        modal_body = _build_modal_chips(items, max_len=80)
        chips.append(
            _build_modal(
                f"+{len(items) - max_items} more",
                f"{title} ({len(items)})",
                modal_body,
                modal_id,
            )
        )
    return (
        "<div>"
        f"<div class='meta-section-title'>{html_lib.escape(title)}</div>"
        f"<div class='meta-chips'>{''.join(chips)}</div>"
        "</div>"
    )


def _format_identified_by(value: Any, max_items: int = 8) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>No names</span>"
    preferred: List[str] = []
    others: List[str] = []
    for item in items:
        label = _extract_label_text(item, max_len=90)
        if not label:
            continue
        is_preferred = False
        if isinstance(item, dict):
            for classified in _coerce_list(item.get("classified_as")):
                if isinstance(classified, dict):
                    cls_label = str(classified.get("_label") or classified.get("label") or "").lower()
                    cls_id = str(classified.get("id") or "")
                    if "preferred" in cls_label or cls_id.endswith("300404670"):
                        is_preferred = True
                        break
                elif isinstance(classified, str) and "300404670" in classified:
                    is_preferred = True
                    break
        if is_preferred:
            preferred.append(label)
        else:
            others.append(label)
    sections = []
    if preferred:
        sections.append(_format_chip_section("Preferred", preferred, max_items, primary=True, modal_key="preferred"))
    if others:
        sections.append(_format_chip_section("Also known as", others, max_items, modal_key="aka"))
    if not sections:
        return "<span class='meta-muted'>No names</span>"
    return f"<div class='meta-stack'>{''.join(sections)}</div>"


def _format_classified_as(value: Any, max_items: int = 10, modal_key: Optional[str] = None) -> str:
    labels = []
    for item in _coerce_list(value):
        if isinstance(item, str) and _is_uri(item):
            label = _shorten_iri(item, max_len=80)
        else:
            label = _extract_label_text(item, max_len=80)
        if label:
            labels.append(label)
    if not labels:
        return "<span class='meta-muted'>None</span>"
    chips = [_chip_html(label) for label in _dedupe_preserve(labels)[:max_items]]
    if len(labels) > max_items:
        modal_id = _make_modal_id(modal_key or "classified", "chips")
        modal_body = _build_modal_chips(labels, max_len=80)
        chips.append(
            _build_modal(
                f"+{len(labels) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'classified_as')} ({len(labels)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-chips'>{''.join(chips)}</div>"


def _conservation_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v)
    if isinstance(value, dict):
        label = _extract_label_text(value, max_len=120)
        return label or str(value)
    return str(value)


def _format_conservation_action_block(item: Any) -> str:
    if not isinstance(item, dict):
        content = _truncate_text(str(item), 240)
        return (
            "<div class='meta-note'>"
            "<div class='meta-note-title'>Conservation Action</div>"
            f"<div class='meta-note-body'>{html_lib.escape(content)}</div>"
            "</div>"
        )

    title_raw = item.get("action") or item.get("status") or "Conservation Action"
    title = _truncate_text(_conservation_value_text(title_raw) or "Conservation Action", 80)

    detail_pairs = []
    for label, key in (
        ("Date", "date"),
        ("Status", "status"),
        ("Agent", "agent"),
        ("Method", "method"),
        ("Materials", "materials"),
        ("Extent", "extent"),
        ("Institution", "institution"),
    ):
        value = item.get(key)
        text = _conservation_value_text(value)
        if text:
            detail_pairs.append(f"{label}: {_truncate_text(text, 180)}")

    notes = []
    for note in _coerce_list(item.get("notes")):
        if isinstance(note, dict):
            content = note.get("content") or note.get("value") or note.get("@value")
            if content:
                notes.append(_truncate_text(str(content), 160))
        elif note:
            notes.append(_truncate_text(str(note), 160))
    if notes:
        detail_pairs.append(f"Notes: {', '.join(notes)}")

    body = "<br>".join(html_lib.escape(text) for text in detail_pairs) if detail_pairs else ""
    if not body:
        body = "<span class='meta-muted'>No details</span>"

    return (
        "<div class='meta-note'>"
        f"<div class='meta-note-title'>{html_lib.escape(title)}</div>"
        f"<div class='meta-note-body'>{body}</div>"
        "</div>"
    )


def _format_conservation_actions(value: Any, max_items: int = 3, modal_key: Optional[str] = None) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    blocks = [_format_conservation_action_block(item) for item in items[:max_items]]
    if not blocks:
        return "<span class='meta-muted'>None</span>"
    if len(items) > max_items:
        modal_id = _make_modal_id(modal_key or "583", "notes")
        full_blocks = [_format_conservation_action_block(item) for item in items]
        modal_body = f"<div class='meta-stack'>{''.join(full_blocks)}</div>"
        blocks.append(
            _build_modal(
                f"+{len(items) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'Conservation Actions')} ({len(items)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-stack'>{''.join(blocks)}</div>"


def _format_notes(value: Any, max_items: int = 3, modal_key: Optional[str] = None) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    note_blocks = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            title = _extract_first_label(item.get("classified_as"), "Note")
            content = item.get("content") or item.get("@value") or item.get("value")
        else:
            title = "Note"
            content = item
        if not content:
            continue
        content_text = _truncate_text(str(content), 220)
        note_blocks.append(
            "<div class='meta-note'>"
            f"<div class='meta-note-title'>{html_lib.escape(str(title))}</div>"
            f"<div class='meta-note-body'>{html_lib.escape(content_text)}</div>"
            "</div>"
        )
    if not note_blocks:
        return "<span class='meta-muted'>None</span>"
    if len(items) > max_items:
        modal_id = _make_modal_id(modal_key or "notes", "notes")
        full_notes = []
        for item in items:
            if isinstance(item, dict):
                title = _extract_first_label(item.get("classified_as"), "Note")
                content = item.get("content") or item.get("@value") or item.get("value")
            else:
                title = "Note"
                content = item
            if not content:
                continue
            content_text = _truncate_text(str(content), 400)
            full_notes.append(
                "<div class='meta-note'>"
                f"<div class='meta-note-title'>{html_lib.escape(str(title))}</div>"
                f"<div class='meta-note-body'>{html_lib.escape(content_text)}</div>"
                "</div>"
            )
        if full_notes:
            notes_html = "".join(full_notes)
        else:
            notes_html = "<span class='meta-muted'>None</span>"
        modal_body = f"<div class='meta-stack'>{notes_html}</div>"
        note_blocks.append(
            _build_modal(
                f"+{len(items) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'notes')} ({len(items)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-stack'>{''.join(note_blocks)}</div>"


def _format_relationships(
    value: Any,
    max_items: int = 8,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    rel_label_by_id: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for entry in _coerce_list(item.get("classified_as")):
            rel_id = None
            rel_label = None
            if isinstance(entry, dict):
                rel_id = entry.get("id") or entry.get("@id")
                rel_label = _extract_pref_label(entry)
                if not rel_label:
                    for key in ("_label", "label", "content", "name", "title"):
                        raw = entry.get(key)
                        if isinstance(raw, str) and raw.strip():
                            rel_label = raw.strip()
                            break
            elif isinstance(entry, str):
                rel_id = entry
            if isinstance(rel_id, str):
                rel_id = remove_fragment(rel_id.strip())
            else:
                rel_id = None
            if rel_id and not rel_label:
                rel_label = _resolve_label_from_lookup(rel_id, label_lookup)
            if rel_id and rel_label:
                rel_label_by_id[rel_id] = str(rel_label).strip()

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        rel_id = None
        for entry in _coerce_list(item.get("classified_as")):
            if isinstance(entry, dict):
                rel_id = entry.get("id") or entry.get("@id")
            elif isinstance(entry, str):
                rel_id = entry
            if isinstance(rel_id, str) and rel_id.strip():
                rel_id = remove_fragment(rel_id.strip())
                break
            rel_id = None
        rel_label = _extract_first_label(item.get("classified_as"), "Related to")
        if rel_id:
            rel_label = rel_label_by_id.get(rel_id) or rel_label
        if _is_uri(rel_label):
            rel_label = _resolve_label_from_lookup(rel_label, label_lookup) or _shorten_iri(
                str(rel_label), max_len=48
            )
        target = (
            item.get("la:relates_to")
            or item.get("relates_to")
            or item.get("related_to")
            or item.get("relates_to_by")
        )
        target_id = None
        if isinstance(target, dict):
            target_id = target.get("id") or target.get("@id")
        elif isinstance(target, str):
            target_id = target
        target_label = None
        if target is not None:
            target_label = _extract_label_text(target, max_len=56)
        if not target_label and target_id is not None:
            target_label = _resolve_label_from_lookup(target_id, label_lookup)
        if target_id is not None:
            lookup_label = _resolve_label_from_lookup(target_id, label_lookup)
            if lookup_label:
                if (
                    not target_label
                    or _label_is_unnamed(target_label, target_id)
                    or (
                        isinstance(target_id, str)
                        and target_label == _shorten_iri(target_id, max_len=56)
                    )
                ):
                    target_label = lookup_label
        if not target_label and isinstance(target_id, str):
            target_label = _shorten_iri(target_id, max_len=56)
        display_label = target_label or "Unknown"
        href = target_id if isinstance(target_id, str) and _is_uri(target_id) else None
        group_key = rel_id or str(rel_label)
        group = grouped.setdefault(group_key, {"label": str(rel_label), "targets": []})
        group["targets"].append((display_label, href))

    if not grouped:
        return "<span class='meta-muted'>None</span>"

    max_groups = min(4, max_items) if max_items else 4
    max_targets = 4
    sections = []
    for idx, group in enumerate(grouped.values()):
        if idx >= max_groups:
            break
        rel_label = group["label"]
        targets = group["targets"]
        chips = []
        for label, href in targets[:max_targets]:
            chips.append(_chip_html(label, href=href))
        if len(targets) > max_targets:
            modal_id = _make_modal_id(rel_label, "targets")
            modal_body = _build_modal_target_chips(targets)
            chips.append(
                _build_modal(
                    f"+{len(targets) - max_targets} more",
                    f"{rel_label} ({len(targets)})",
                    modal_body,
                    modal_id,
                )
            )
        title = f"{rel_label} ({len(targets)})"
        sections.append(
            "<div class='meta-rel-group'>"
            f"<div class='meta-section-title'>{html_lib.escape(title)}</div>"
            f"<div class='meta-chips'>{''.join(chips)}</div>"
            "</div>"
        )

    if len(grouped) > max_groups:
        modal_id = _make_modal_id("relationships", "groups")
        full_groups = []
        for group in grouped.values():
            rel_label = group["label"]
            targets = group["targets"]
            chips = _build_modal_target_chips(targets)
            full_groups.append(
                "<div class='meta-rel-group'>"
                f"<div class='meta-section-title'>{html_lib.escape(rel_label)} ({len(targets)})</div>"
                f"{chips}"
                "</div>"
            )
        modal_body = f"<div class='meta-stack'>{''.join(full_groups)}</div>"
        sections.append(
            _build_modal(
                f"+{len(grouped) - max_groups} types more",
                "Relationships",
                modal_body,
                modal_id,
            )
        )

    return f"<div class='meta-stack'>{''.join(sections)}</div>"


def _format_uri_list(
    value: Any,
    max_items: int = 8,
    modal_key: Optional[str] = None,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    items = _coerce_list(value)
    if not items:
        return "<span class='meta-muted'>None</span>"
    chips = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            ident = item.get("id") or item.get("@id")
            label = _resolve_label_from_lookup(item, label_lookup) or _extract_label_text(item, max_len=60)
            if not label and isinstance(ident, str):
                label = _shorten_iri(ident, max_len=60)
        else:
            ident = item
            label = _resolve_label_from_lookup(item, label_lookup)
            if not label:
                if isinstance(item, str) and _is_uri(item):
                    label = _shorten_iri(item, max_len=60)
                else:
                    label = _extract_label_text(item, max_len=60) or (ident if isinstance(ident, str) else "")
        label = _truncate_text(str(label), 60)
        href = ident if isinstance(ident, str) else None
        chips.append(_chip_html(label, href=href))
    if len(items) > max_items:
        modal_id = _make_modal_id(modal_key or "uris", "uris")
        modal_body = _build_modal_chips(items, max_len=70)
        chips.append(
            _build_modal(
                f"+{len(items) - max_items} more",
                f"{_humanize_meta_key(modal_key or 'uris')} ({len(items)})",
                modal_body,
                modal_id,
            )
        )
    return f"<div class='meta-chips'>{''.join(chips)}</div>"


def _metadata_rows(metadata: Dict[str, Any], skip_keys: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    rows = []
    skip_keys = skip_keys or set()
    for key, value in metadata.items():
        if key in skip_keys:
            continue
        rows.append({"Property": str(key), "Value": _summarize_value(value)})
    return rows


def _serialize_embeddings(embeddings: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[float]]]:
    if not embeddings:
        return None
    serialized: Dict[str, List[float]] = {}
    for key, vec in embeddings.items():
        if vec is None:
            continue
        if isinstance(vec, np.ndarray):
            vec_list = vec.tolist()
        elif isinstance(vec, list):
            vec_list = vec
        else:
            try:
                vec_list = list(vec)
            except TypeError:
                continue
        try:
            serialized[str(key)] = [float(x) for x in vec_list]
        except (TypeError, ValueError):
            continue
    return serialized or None


def _serialize_centrality(
    centrality: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[Dict[str, Dict[str, float]]]:
    if not centrality:
        return None
    serialized: Dict[str, Dict[str, float]] = {}
    for node_id, metrics in centrality.items():
        if not isinstance(metrics, dict):
            continue
        safe_metrics: Dict[str, float] = {}
        for metric_name, metric_value in metrics.items():
            try:
                safe_metrics[str(metric_name)] = float(metric_value)
            except (TypeError, ValueError):
                continue
        if safe_metrics:
            serialized[str(node_id)] = safe_metrics
    return serialized or None


def _serialize_graph_data(graph_data: GraphData) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "types": list(n.types or []),
                "metadata": n.metadata,
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "relationship": e.relationship,
                        "inferred": bool(getattr(e, "inferred", False)),
                        "weight": e.weight,
                    }
                    for e in n.edges
                ],
            }
            for n in graph_data.nodes
        ]
    }


def _deserialize_graph_data(payload: Dict[str, Any]) -> GraphData:
    nodes_payload = payload.get("nodes", []) if isinstance(payload, dict) else []
    nodes: List[Node] = []
    for raw in nodes_payload:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id", "")).strip()
        label = raw.get("label") or node_id
        raw_types = raw.get("types") or []
        if not isinstance(raw_types, list):
            raw_types = [raw_types]
        metadata = raw.get("metadata", {})
        edges = []
        for edge in raw.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("source", node_id))
            tgt = str(edge.get("target", "")).strip()
            rel = str(edge.get("relationship", "")).strip()
            inferred = bool(edge.get("inferred", False))
            weight = edge.get("weight", None)
            if weight is not None:
                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    weight = None
            if src and tgt and rel:
                edges.append(
                    Edge(
                        source=src,
                        target=tgt,
                        relationship=rel,
                        inferred=inferred,
                        weight=weight,
                    )
                )
        if node_id:
            nodes.append(Node(id=node_id, label=label, types=raw_types, metadata=metadata, edges=edges))
    return GraphData(nodes=nodes)


def _format_meta_value_html(
    value: Any,
    key: Optional[str] = None,
    max_items: int = 6,
    max_len: int = 140,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    if key in {"583", "conservationActions", "conservation_actions"}:
        return _format_conservation_actions(value, modal_key=key)
    if key in {"born", "died"}:
        return _format_event_details(value)
    if key in {"dateOfBirth", "dateOfDeath", "date_of_birth", "date_of_death", "birth", "death"}:
        date_text = _extract_date_text(value)
        if date_text:
            return html_lib.escape(date_text)
    if key and any(token in key.lower() for token in ("date", "birth", "death")):
        date_text = _extract_date_text(value)
        if date_text:
            return html_lib.escape(date_text)
    if key in {"creationDate", "startDate", "endDate"}:
        return _format_time_list(value, max_items=max_items, modal_key=key)
    if key in {"creator", "contributor", "draftsman", "author", "editor", "illustrator"}:
        return _format_role_list(value, max_items=max_items, modal_key=key, label_lookup=label_lookup)
    if key in {"description", "title"}:
        return _format_lang_text(value, max_len=240 if key == "description" else 180)
    if key == "identified_by":
        return _format_identified_by(value)
    if key == "classified_as":
        return _format_classified_as(value, max_items=max_items, modal_key=key)
    if key in {"referred_to_by", "subject_of"}:
        return _format_notes(value, modal_key=key)
    if key in {"notes", "note"}:
        return _format_notes(value, modal_key=key)
    if key in {"la:related_from_by", "la:related_to_by"}:
        return _format_relationships(value, label_lookup=label_lookup)
    if key in {"sameAs", "rdfs:seeAlso", "@context", "skos:inScheme"}:
        return _format_uri_list(value, modal_key=key, label_lookup=label_lookup)
    if key in {"timespan", "took_place_at", "language"}:
        if key == "timespan":
            date_text = _format_timespan_details(value)
            if date_text:
                return html_lib.escape(date_text)
        return _format_classified_as(value, max_items=max_items, modal_key=key)
    if value is None:
        return "<span class='meta-muted'>None</span>"
    if isinstance(value, list):
        if not value:
            return "<span class='meta-muted'>Empty list</span>"
        chips = []
        for item in value[:max_items]:
            lookup_label = _resolve_label_from_lookup(item, label_lookup)
            label = lookup_label or _extract_label_text(item, max_len=60)
            href = None
            if isinstance(item, dict):
                ident = item.get("id") or item.get("@id")
                href = ident if _is_uri(ident) else None
            elif isinstance(item, str) and _is_uri(item):
                href = item
            if label:
                if isinstance(item, str) and _is_uri(item) and label == item:
                    label = _shorten_iri(item, max_len=60)
                chips.append(_chip_html(label, href=href))
            else:
                chip_text = _summarize_value(item, max_items=3, max_len=60)
                chips.append(f"<span class='meta-chip'>{html_lib.escape(str(chip_text))}</span>")
        if len(value) > max_items:
            modal_id = _make_modal_id(key or "list", "list")
            modal_body = _build_modal_chips(value, max_len=70)
            chips.append(
                _build_modal(
                    f"+{len(value) - max_items} more",
                    f"{_humanize_meta_key(key or 'Items')} ({len(value)})",
                    modal_body,
                    modal_id,
                )
            )
        return f"<div class='meta-chips'>{''.join(chips)}</div>"
    if isinstance(value, dict):
        if not value:
            return "<span class='meta-muted'>Empty object</span>"
        label = _extract_label_text(value, max_len=80)
        if label and set(value.keys()).issubset({"id", "@id", "type", "@type", "_label", "label", "content"}):
            ident = value.get("id") or value.get("@id")
            if isinstance(ident, str) and _is_uri(ident):
                return _format_link(ident, label)
            return html_lib.escape(label)
        pairs = []
        items = list(value.items())
        for subkey, subvalue in items[:max_items]:
            rendered = _summarize_value(subvalue, max_items=3, max_len=60)
            pairs.append(
                "<div class='meta-kv'>"
                f"<span>{html_lib.escape(_humanize_meta_key(subkey))}</span>"
                f"<strong>{html_lib.escape(str(rendered))}</strong>"
                "</div>"
            )
        if len(items) > max_items:
            pairs.append(
                "<div class='meta-kv'>"
                "<span class='meta-muted'>More</span>"
                f"<strong class='meta-muted'>+{len(items) - max_items}</strong>"
                "</div>"
            )
        return f"<div class='meta-kv-list'>{''.join(pairs)}</div>"
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith(("http://", "https://")):
            label = _truncate_text(cleaned, 46)
            return (
                f"<a class='meta-link' href='{html_lib.escape(cleaned)}' "
                f"target='_blank' rel='noopener noreferrer'>{html_lib.escape(label)}</a>"
            )
        return html_lib.escape(_truncate_text(cleaned, max_len))
    if isinstance(value, (int, float, bool)):
        return html_lib.escape(str(value))
    return html_lib.escape(_truncate_text(value, max_len))


def _render_metadata_overview(
    metadata: Dict[str, Any],
    skip_keys: Optional[Set[str]] = None,
    max_cards: int = 12,
    label_lookup: Optional[Dict[str, str]] = None,
) -> str:
    if not metadata:
        return ""
    skip_keys = skip_keys or set()
    items = [(k, v) for k, v in metadata.items() if k not in skip_keys]
    if not items:
        return ""
    cleaned_items = []
    for key, value in items:
        if value is None or value == "" or value == {} or value == []:
            continue
        cleaned_items.append((key, value))
    if not cleaned_items:
        return ""
    items = sorted(
        cleaned_items,
        key=lambda item: (
            _META_KEY_PRIORITY.get(str(item[0]), 50),
            str(item[0]).lower(),
        ),
    )
    cards = []
    for key, value in items[:max_cards]:
        value_html = _format_meta_value_html(value, key=str(key), label_lookup=label_lookup)
        cards.append(
            "<div class='meta-card'>"
            f"<div class='meta-key'>{html_lib.escape(_humanize_meta_key(key))}</div>"
            f"<div class='meta-value'>{value_html}</div>"
            "</div>"
        )
    remaining = len(items) - max_cards
    if remaining > 0:
        cards.append(
            "<div class='meta-card meta-card-more'>"
            "<div class='meta-key'>More</div>"
            f"<div class='meta-value'><span class='meta-muted'>+{remaining} more in Raw</span></div>"
            "</div>"
        )
    return f"<div class='meta-grid'>{''.join(cards)}</div>"


def _render_type_badges(types: List[str], palette: Optional[Dict[str, str]] = None) -> str:
    if not types:
        types = ["Unknown"]
    palette = palette or CONFIG["NODE_TYPE_COLORS"]
    badges = []
    for t in types:
        canon = canonical_type(t)
        color = palette.get(canon) or CONFIG["NODE_TYPE_COLORS"].get(canon, CONFIG["DEFAULT_NODE_COLOR"])
        label_color = _pick_label_color(color)
        badge = (
            "<span style='display:inline-block; padding:4px 8px; border-radius:999px; "
            f"background:{color}; color:{label_color}; font-size:12px; margin-right:6px; "
            "border:1px solid rgba(15, 23, 42, 0.12);'>"
            f"{html_lib.escape(canon)}"
            "</span>"
        )
        badges.append(badge)
    return "".join(badges)


def _local_name(t: str) -> str:
    """Return the local name from an IRI or CURIE (after # or / or :)."""
    if not isinstance(t, str) or not t:
        return "Unknown"
    s = t.strip()
    if s.startswith("http://") or s.startswith("https://"):
        s = s.rstrip("/#")
        if "#" in s:
            s = s.split("#")[-1]
        else:
            s = s.split("/")[-1]
    if ":" in s and not s.startswith(("http://", "https://")):
        s = s.split(":", 1)[-1]
    return s or "Unknown"


def canonical_type(t: str) -> str:
    """
    Normalize a raw type (IRI/CURIE/local) into one of the legend keys in
    CONFIG['NODE_TYPE_COLORS']; otherwise return 'Unknown'.
    """
    name = _local_name(t)
    mapped = TYPE_SYNONYMS.get(name, name)
    return mapped if mapped in CONFIG["NODE_TYPE_COLORS"] else "Unknown"


def _shorten_iri(iri: str, max_len: int = 80) -> str:
    if not isinstance(iri, str):
        return str(iri)
    for ns, prefix in NAMESPACE_PREFIXES.items():
        if iri.startswith(ns):
            return f"{prefix}:{iri[len(ns):]}"
    if len(iri) <= max_len:
        return iri
    local = _local_name(iri)
    if local and local != "Unknown":
        prefix_len = max_len - len(local) - 4
        if prefix_len < 5:
            return iri[: max_len - 3] + "..."
        return f"{local} ({iri[:prefix_len]}...)"
    return iri[: max_len - 3] + "..."


def _extract_inferred_types(metadata: Any) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    inferred = metadata.get("inferredTypes")
    if inferred is None:
        return []
    if isinstance(inferred, list):
        return [str(t) for t in inferred]
    return [str(inferred)]


def _type_uri(t: Any) -> URIRef:
    if isinstance(t, URIRef):
        return t
    t_str = str(t).strip()
    if t_str.startswith(("http://", "https://")):
        return URIRef(t_str)
    if ":" in t_str and not t_str.startswith(("http://", "https://")):
        prefix, local = t_str.split(":", 1)
        if prefix == "rdf":
            return RDF[local]
        if prefix == "rdfs":
            return RDFS[local]
    return EX[t_str]


def _predicate_uri(rel: Any) -> URIRef:
    if isinstance(rel, URIRef):
        return rel
    rel_str = str(rel).strip()
    if rel_str.startswith(("http://", "https://")):
        return URIRef(rel_str)
    if ":" in rel_str and not rel_str.startswith(("http://", "https://")):
        prefix, local = rel_str.split(":", 1)
        if prefix == "rdfs":
            return RDFS[local]
        if prefix == "rdf":
            return RDF[local]
    rel_lower = rel_str.replace(" ", "").replace("_", "").replace("-", "").lower()
    for suffix, uri in RDFS_PREDICATE_SUFFIXES.items():
        if rel_lower.endswith(suffix):
            return uri
    return EX[rel_str]


def slugify_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return value or "network-graph"
_META_MODAL_COUNTER = itertools.count(1)
