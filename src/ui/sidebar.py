"""Sidebar logic and session state initialization."""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
import requests
import streamlit as st
from rdflib import Graph as RDFGraph

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    boto3_installed = True
except ImportError:  # pragma: no cover - optional dependency
    boto3_installed = False
    BotoCoreError = ClientError = Exception

from src.config import CONFIG
from src.data_processing import (
    apply_rdfs_reasoning,
    compute_centrality_measures,
    compute_anomaly_flags,
    compute_node_roles,
    compute_probabilistic_graph_embeddings,
    compute_rdfs_signature,
    convert_graph_data_to_rdf,
    dereference_uri,
    enhance_graph_data_from_triples,
    find_search_nodes,
    collapse_same_as_nodes,
    load_data_from_sparql,
    link_oclc_creators,
    node2vec_installed,
    owlrl_installed,
    parse_entities_from_contents,
    parse_marc_content,
    parse_ris_content,
    process_uploaded_file,
    run_sparql_query,
    suggest_ontologies,
    validate_with_shacl,
)
from src.models import Edge, GraphData, Node
from src.utils import (
    _deserialize_graph_data,
    _extract_inferred_types,
    _label_is_unnamed,
    _serialize_centrality,
    _serialize_embeddings,
    _serialize_graph_data,
    _shorten_iri,
    canonical_type,
    refresh_label_index,
    slugify_filename,
)


@dataclass
class SidebarState:
    filtered_nodes: Optional[Set[str]]
    effective_node_animations: List[str]
    community_detection: bool


_HEX_COLOR_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _normalize_hex_color(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if not _HEX_COLOR_RE.match(text):
        return None
    return f"#{text}"


def _read_secret(key: str) -> str:
    return str(os.getenv(key, "") or "").strip()


def _spaces_endpoint(region: str, endpoint_override: str) -> str:
    endpoint = endpoint_override.strip() if endpoint_override else ""
    if endpoint:
        return endpoint.rstrip("/")
    cleaned_region = (region or "").strip()
    if not cleaned_region:
        return ""
    return f"https://{cleaned_region}.digitaloceanspaces.com"


def _normalize_spaces_prefix(prefix: str) -> str:
    cleaned = (prefix or "").strip().strip("/")
    return f"{cleaned}/" if cleaned else ""


def _spaces_client(access_key: str, secret_key: str, region: str, endpoint: str):
    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _spaces_list_workspaces(client, bucket: str, prefix: str) -> List[Dict[str, object]]:
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix or "")
    contents = response.get("Contents", []) or []
    items: List[Dict[str, object]] = []
    for entry in contents:
        key = entry.get("Key") or ""
        if not key.endswith(".json"):
            continue
        items.append(
            {
                "key": key,
                "last_modified": entry.get("LastModified"),
                "size": entry.get("Size"),
            }
        )
    return items


def _spaces_download_workspace(client, bucket: str, key: str) -> str:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None:
        return ""
    return body.read().decode("utf-8")


def _spaces_upload_workspace(client, bucket: str, key: str, payload: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )


def _coerce_palette(palette: object, defaults: Dict[str, str]) -> Dict[str, str]:
    merged = dict(defaults)
    if not isinstance(palette, dict):
        return merged
    for key, raw in palette.items():
        normalized = _normalize_hex_color(raw)
        if normalized:
            merged[str(key)] = normalized
    return merged


def _coerce_background(background: object, defaults: Dict[str, str]) -> Dict[str, str]:
    merged = dict(defaults)
    if not isinstance(background, dict):
        return merged
    for key in defaults.keys():
        normalized = _normalize_hex_color(background.get(key))
        if normalized:
            merged[key] = normalized
    return merged


def _coerce_color_list(values: object, defaults: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(defaults)
    cleaned: List[str] = []
    for value in values:
        normalized = _normalize_hex_color(value)
        if normalized:
            cleaned.append(normalized)
    return cleaned if cleaned else list(defaults)


def _coerce_string_list(values: object, defaults: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(defaults)
    cleaned: List[str] = []
    for value in values:
        if value is None:
            cleaned.append("")
        else:
            cleaned.append(str(value))
    return cleaned if cleaned else list(defaults)


def _coerce_lock_map(values: object) -> Dict[int, str]:
    if not isinstance(values, dict):
        return {}
    cleaned: Dict[int, str] = {}
    for raw_key, raw_value in values.items():
        try:
            idx = int(raw_key)
        except (TypeError, ValueError):
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            cleaned[idx] = value
    return cleaned


def _coerce_label_map(values: object) -> Dict[str, str]:
    if not isinstance(values, dict):
        return {}
    cleaned: Dict[str, str] = {}
    for raw_key, raw_value in values.items():
        if raw_value is None:
            continue
        label = str(raw_value).strip()
        if label:
            cleaned[str(raw_key)] = label
    return cleaned


def _palette_key(prefix: str, name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return f"{prefix}_{safe}"


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    hue = hue % 1.0
    saturation = max(0.0, min(1.0, saturation))
    lightness = max(0.0, min(1.0, lightness))
    if saturation == 0:
        channel = int(round(lightness * 255))
        return f"#{channel:02X}{channel:02X}{channel:02X}"

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = lightness * (1 + saturation) if lightness < 0.5 else lightness + saturation - lightness * saturation
    p = 2 * lightness - q
    r = hue_to_rgb(p, q, hue + 1 / 3)
    g = hue_to_rgb(p, q, hue)
    b = hue_to_rgb(p, q, hue - 1 / 3)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _generate_community_color(index: int) -> str:
    golden_ratio = 0.61803398875
    hue = (index * golden_ratio) % 1.0
    return _hsl_to_hex(hue, 0.55, 0.55)


def _sync_community_widget_state(
    palette: List[str], labels: List[str], locks: Dict[int, str]
) -> None:
    for idx, color in enumerate(palette):
        st.session_state[f"community_color_{idx}"] = color
        label_value = labels[idx] if idx < len(labels) else f"Community {idx + 1}"
        st.session_state[f"community_label_{idx}"] = label_value
        st.session_state[f"community_lock_{idx}"] = locks.get(idx, "")
    # Remove stale widget keys for deleted slots
    for key in list(st.session_state.keys()):
        if not key.startswith("community_"):
            continue
        if key.startswith("community_color_") or key.startswith("community_label_") or key.startswith(
            "community_lock_"
        ):
            try:
                idx = int(key.split("_")[-1])
            except ValueError:
                continue
            if idx >= len(palette):
                st.session_state.pop(key, None)


def init_session_state() -> None:
    if "node_positions" not in st.session_state:
        st.session_state.node_positions = {}
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "selected_relationships" not in st.session_state:
        st.session_state.selected_relationships = list(CONFIG["RELATIONSHIP_CONFIG"].keys())
    if "search_term" not in st.session_state:
        st.session_state.search_term = ""
    if "search_nodes" not in st.session_state:
        st.session_state.search_nodes = []
    if "show_labels" not in st.session_state:
        st.session_state.show_labels = True
    if "smart_labels" not in st.session_state:
        st.session_state.smart_labels = True
    if "label_zoom_threshold" not in st.session_state:
        st.session_state.label_zoom_threshold = 1.1
    if "sparql_query" not in st.session_state:
        st.session_state.sparql_query = ""
    if "filtered_types" not in st.session_state:
        st.session_state.filtered_types = []
    if "filtered_inferred_types" not in st.session_state:
        st.session_state.filtered_inferred_types = []
    if "enable_physics" not in st.session_state:
        st.session_state.enable_physics = True
    if "reduce_motion" not in st.session_state:
        st.session_state.reduce_motion = False
    if "focus_context" not in st.session_state:
        st.session_state.focus_context = True
    if "edge_semantics" not in st.session_state:
        st.session_state.edge_semantics = True
    if "type_icons" not in st.session_state:
        st.session_state.type_icons = True
    if "performance_mode" not in st.session_state:
        st.session_state.performance_mode = False
    if "performance_mode_applied" not in st.session_state:
        st.session_state.performance_mode_applied = False
    if "performance_backup" not in st.session_state:
        st.session_state.performance_backup = {}
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = GraphData(nodes=[])
    if "id_to_label" not in st.session_state:
        st.session_state.id_to_label = {}
    if "node_type_colors" not in st.session_state:
        st.session_state.node_type_colors = CONFIG["NODE_TYPE_COLORS"].copy()
    if "relationship_colors" not in st.session_state:
        st.session_state.relationship_colors = CONFIG["RELATIONSHIP_CONFIG"].copy()
    if "graph_background" not in st.session_state:
        st.session_state.graph_background = CONFIG.get("GRAPH_BACKGROUND", {}).copy()
    if "community_colors" not in st.session_state:
        st.session_state.community_colors = CONFIG.get("COMMUNITY_COLORS", []).copy()
    if "community_labels" not in st.session_state:
        st.session_state.community_labels = [
            f"Community {idx + 1}" for idx in range(len(st.session_state.community_colors))
        ]
    if "community_locks" not in st.session_state:
        st.session_state.community_locks = {}
    if "community_lock_labels" not in st.session_state:
        st.session_state.community_lock_labels = {}
    if "physics_params" not in st.session_state:
        st.session_state.physics_params = CONFIG["PHYSICS_DEFAULTS"].copy()
    if "modal_action" not in st.session_state:
        st.session_state.modal_action = None
    if "modal_node" not in st.session_state:
        st.session_state.modal_node = None
    if "centrality_measures" not in st.session_state:
        st.session_state.centrality_measures = None
    if "spaces_workspace_list" not in st.session_state:
        st.session_state.spaces_workspace_list = None
    if "node_roles" not in st.session_state:
        st.session_state.node_roles = None
    if "role_signature" not in st.session_state:
        st.session_state.role_signature = None
    if "role_filters" not in st.session_state:
        st.session_state.role_filters = []
    if "node_anomalies" not in st.session_state:
        st.session_state.node_anomalies = None
    if "anomaly_signature" not in st.session_state:
        st.session_state.anomaly_signature = None
    if "anomaly_filters" not in st.session_state:
        st.session_state.anomaly_filters = []
    if "shortest_path" not in st.session_state:
        st.session_state.shortest_path = None
    if "property_filter" not in st.session_state:
        st.session_state.property_filter = {"property": "", "value": ""}
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "annotation_html" not in st.session_state:
        st.session_state.annotation_html = {}
    if "graph_embeddings" not in st.session_state:
        st.session_state.graph_embeddings = None
    if "textual_semantic_embeddings" not in st.session_state:
        st.session_state.textual_semantic_embeddings = None
    if "export_title" not in st.session_state:
        st.session_state.export_title = "Linked Data Explorer"
    if "export_title_draft" not in st.session_state:
        st.session_state.export_title_draft = st.session_state.export_title
    if "upload_errors" not in st.session_state:
        st.session_state.upload_errors = []
    if "workspace_name" not in st.session_state:
        st.session_state.workspace_name = "Workspace"
    if "workspace_name_input" not in st.session_state:
        st.session_state.workspace_name_input = st.session_state.workspace_name
    if "workspace_name_pending" not in st.session_state:
        st.session_state.workspace_name_pending = None
    if "ris_import_stats" not in st.session_state:
        st.session_state.ris_import_stats = None
    if "marc_import_stats" not in st.session_state:
        st.session_state.marc_import_stats = None
    if "oclc_link_stats" not in st.session_state:
        st.session_state.oclc_link_stats = None
    if "oclc_collapse_stats" not in st.session_state:
        st.session_state.oclc_collapse_stats = None
    if "oclc_base_graph" not in st.session_state:
        st.session_state.oclc_base_graph = None
    if "oclc_base_labels" not in st.session_state:
        st.session_state.oclc_base_labels = None
    if "oclc_base_signature" not in st.session_state:
        st.session_state.oclc_base_signature = None
    if "oclc_collapsed_graph" not in st.session_state:
        st.session_state.oclc_collapsed_graph = None
    if "oclc_collapsed_labels" not in st.session_state:
        st.session_state.oclc_collapsed_labels = None
    if "oclc_collapsed_signature" not in st.session_state:
        st.session_state.oclc_collapsed_signature = None
    if "collapse_oclc_last" not in st.session_state:
        st.session_state.collapse_oclc_last = None
    if "motion_intensity" not in st.session_state:
        st.session_state.motion_intensity = 45
    if "node_animations" not in st.session_state:
        legacy_animation = st.session_state.get("node_animation", "none")
        if isinstance(legacy_animation, list):
            st.session_state.node_animations = legacy_animation
        elif isinstance(legacy_animation, str):
            st.session_state.node_animations = [legacy_animation] if legacy_animation else ["none"]
        else:
            st.session_state.node_animations = ["none"]
    if "node_animation_strength" not in st.session_state:
        st.session_state.node_animation_strength = 30
    if "community_detection" not in st.session_state:
        st.session_state.community_detection = False
    if "arc_map_zoom" not in st.session_state:
        st.session_state.arc_map_zoom = 2.0
    if "arc_map_pitch" not in st.session_state:
        st.session_state.arc_map_pitch = 35


def _build_workspace_snapshot() -> Dict[str, object]:
    annotations_snapshot = dict(st.session_state.annotations or {})
    annotations_html_snapshot = dict(st.session_state.annotation_html or {})
    for node in st.session_state.graph_data.nodes:
        if not isinstance(node.metadata, dict):
            continue
        if node.id not in annotations_snapshot and node.metadata.get("annotation"):
            annotations_snapshot[node.id] = str(node.metadata.get("annotation") or "")
        if node.id not in annotations_html_snapshot and node.metadata.get("annotation_html"):
            annotations_html_snapshot[node.id] = str(node.metadata.get("annotation_html") or "")
        if node.id in annotations_snapshot and node.id not in annotations_html_snapshot:
            escaped = html_lib.escape(str(annotations_snapshot[node.id])).replace("\n", "<br>")
            annotations_html_snapshot[node.id] = escaped
    label_snapshot = refresh_label_index(
        st.session_state.graph_data,
        st.session_state.id_to_label,
    )
    base_labels = st.session_state.get("oclc_base_labels")
    if isinstance(base_labels, dict):
        for node_id, label in base_labels.items():
            if not node_id or _label_is_unnamed(label, node_id):
                continue
            existing = label_snapshot.get(node_id)
            if not existing or _label_is_unnamed(existing, node_id):
                label_snapshot[node_id] = label
    st.session_state.id_to_label = label_snapshot
    return {
        "version": 1,
        "saved_at": f"{datetime.utcnow().isoformat()}Z",
        "graph": _serialize_graph_data(st.session_state.graph_data),
        "id_to_label": label_snapshot,
        "node_positions": st.session_state.node_positions,
        "annotations": annotations_snapshot,
        "annotations_html": annotations_html_snapshot,
        "selection": {
            "selected_node": st.session_state.selected_node,
            "selected_relationships": st.session_state.selected_relationships,
            "shortest_path": st.session_state.shortest_path,
        },
        "filters": {
            "filtered_types": st.session_state.filtered_types,
            "filtered_inferred_types": st.session_state.filtered_inferred_types,
            "property_filter": st.session_state.property_filter,
            "search_term": st.session_state.search_term,
            "sparql_query": st.session_state.sparql_query,
            "role_filters": st.session_state.role_filters,
            "anomaly_filters": st.session_state.anomaly_filters,
        },
        "view": {
            "show_labels": st.session_state.show_labels,
            "smart_labels": st.session_state.smart_labels,
            "label_zoom_threshold": st.session_state.label_zoom_threshold,
            "reduce_motion": st.session_state.reduce_motion,
            "motion_intensity": st.session_state.motion_intensity,
            "node_animations": st.session_state.node_animations,
            "node_animation_strength": st.session_state.node_animation_strength,
            "enable_physics": st.session_state.enable_physics,
            "focus_context": st.session_state.focus_context,
            "edge_semantics": st.session_state.edge_semantics,
            "community_detection": st.session_state.get("community_detection", False),
            "physics_preset": st.session_state.get("physics_preset", "Default (Balanced)"),
            "physics_params": st.session_state.physics_params,
            "performance_mode": st.session_state.performance_mode,
            "performance_mode_applied": st.session_state.performance_mode_applied,
            "performance_backup": st.session_state.performance_backup,
        },
        "meta": {
            "export_title": st.session_state.export_title,
            "workspace_name": st.session_state.workspace_name,
        },
        "appearance": {
            "node_type_colors": st.session_state.node_type_colors,
            "relationship_colors": st.session_state.relationship_colors,
            "graph_background": st.session_state.graph_background,
            "type_icons": st.session_state.type_icons,
            "community_colors": st.session_state.community_colors,
            "community_labels": st.session_state.community_labels,
            "community_locks": st.session_state.community_locks,
            "community_lock_labels": st.session_state.community_lock_labels,
        },
        "analytics": {
            "centrality_measures": _serialize_centrality(st.session_state.centrality_measures),
            "graph_embeddings": _serialize_embeddings(st.session_state.graph_embeddings),
            "textual_semantic_embeddings": _serialize_embeddings(
                st.session_state.textual_semantic_embeddings
            ),
        },
    }


def _restore_workspace(snapshot: Dict[str, object]) -> None:
    graph_payload = snapshot.get("graph") or snapshot.get("graph_data") or {}
    graph_data = _deserialize_graph_data(graph_payload)
    id_to_label = snapshot.get("id_to_label") or {}
    id_to_label = refresh_label_index(graph_data, id_to_label)

    st.session_state.graph_data = graph_data
    st.session_state.id_to_label = id_to_label
    st.session_state.node_roles = None
    st.session_state.role_signature = None
    st.session_state.node_anomalies = None
    st.session_state.anomaly_signature = None

    annotations = snapshot.get("annotations") or {}
    if not isinstance(annotations, dict):
        annotations = {}
    annotations_html = snapshot.get("annotations_html") or {}
    if not isinstance(annotations_html, dict):
        annotations_html = {}
    st.session_state.annotations = annotations
    st.session_state.annotation_html = annotations_html
    for node in st.session_state.graph_data.nodes:
        if not isinstance(node.metadata, dict):
            node.metadata = {}
        if node.id in annotations:
            node.metadata["annotation"] = annotations[node.id]
        if node.id in annotations_html:
            node.metadata["annotation_html"] = annotations_html[node.id]
        elif node.id in annotations:
            escaped = html_lib.escape(str(annotations[node.id])).replace("\n", "<br>")
            node.metadata["annotation_html"] = escaped
        if node.id not in annotations and node.metadata.get("annotation"):
            annotations[node.id] = str(node.metadata.get("annotation") or "")
        if node.id not in annotations_html and node.metadata.get("annotation_html"):
            annotations_html[node.id] = str(node.metadata.get("annotation_html") or "")
        if node.id in annotations and node.id not in annotations_html:
            escaped = html_lib.escape(str(annotations[node.id])).replace("\n", "<br>")
            annotations_html[node.id] = escaped
    st.session_state.annotations = annotations
    st.session_state.annotation_html = annotations_html

    node_positions = snapshot.get("node_positions") or {}
    if not isinstance(node_positions, dict):
        node_positions = {}
    st.session_state.node_positions = node_positions

    selection = snapshot.get("selection") or {}
    st.session_state.selected_node = selection.get("selected_node")
    selected_relationships = selection.get("selected_relationships")
    if not isinstance(selected_relationships, list) or not selected_relationships:
        selected_relationships = list(CONFIG["RELATIONSHIP_CONFIG"].keys())
    st.session_state.selected_relationships = selected_relationships
    shortest_path = selection.get("shortest_path")
    if isinstance(shortest_path, list):
        st.session_state.shortest_path = shortest_path
    else:
        st.session_state.shortest_path = None

    filters = snapshot.get("filters") or {}
    filtered_types = filters.get("filtered_types", [])
    if not isinstance(filtered_types, list):
        filtered_types = [filtered_types]
    st.session_state.filtered_types = filtered_types
    filtered_inferred_types = filters.get("filtered_inferred_types", [])
    if not isinstance(filtered_inferred_types, list):
        filtered_inferred_types = [filtered_inferred_types]
    st.session_state.filtered_inferred_types = filtered_inferred_types
    property_filter = filters.get("property_filter")
    if not isinstance(property_filter, dict):
        property_filter = {"property": "", "value": ""}
    st.session_state.property_filter = property_filter
    st.session_state.search_term = filters.get("search_term", "")
    st.session_state.sparql_query = filters.get("sparql_query", "")
    role_filters = filters.get("role_filters", [])
    if not isinstance(role_filters, list):
        role_filters = [role_filters]
    st.session_state.role_filters = role_filters
    anomaly_filters = filters.get("anomaly_filters", [])
    if not isinstance(anomaly_filters, list):
        anomaly_filters = [anomaly_filters]
    st.session_state.anomaly_filters = anomaly_filters

    meta = snapshot.get("meta") or {}
    workspace_name = meta.get("workspace_name")
    if isinstance(workspace_name, str) and workspace_name.strip():
        cleaned = workspace_name.strip()
        st.session_state.workspace_name = cleaned
        st.session_state.workspace_name_pending = cleaned
    if st.session_state.search_term.strip() and st.session_state.graph_data.nodes:
        st.session_state.search_nodes = find_search_nodes(
            st.session_state.graph_data, st.session_state.search_term
        )
    else:
        st.session_state.search_nodes = []

    view = snapshot.get("view") or {}
    st.session_state.show_labels = view.get("show_labels", True)
    st.session_state.smart_labels = view.get("smart_labels", st.session_state.smart_labels)
    st.session_state.label_zoom_threshold = view.get(
        "label_zoom_threshold", st.session_state.label_zoom_threshold
    )
    st.session_state.reduce_motion = view.get("reduce_motion", False)
    st.session_state.motion_intensity = view.get("motion_intensity", 45)
    node_animations = view.get("node_animations", ["none"])
    if isinstance(node_animations, str):
        node_animations = [node_animations]
    elif not isinstance(node_animations, list):
        node_animations = ["none"]
    st.session_state.node_animations = node_animations
    st.session_state.node_animation_strength = view.get("node_animation_strength", 30)
    st.session_state.enable_physics = view.get("enable_physics", True)
    st.session_state.focus_context = view.get("focus_context", st.session_state.focus_context)
    st.session_state.edge_semantics = view.get("edge_semantics", st.session_state.edge_semantics)
    st.session_state.community_detection = view.get("community_detection", False)
    st.session_state.physics_params = view.get("physics_params", CONFIG["PHYSICS_DEFAULTS"].copy())
    st.session_state.physics_preset = view.get("physics_preset", "Default (Balanced)")
    st.session_state.performance_mode = view.get("performance_mode", False)
    st.session_state.performance_mode_applied = view.get(
        "performance_mode_applied", st.session_state.performance_mode
    )
    st.session_state.performance_backup = view.get("performance_backup", {})

    appearance = snapshot.get("appearance") or {}
    node_type_colors = appearance.get("node_type_colors") or snapshot.get("node_type_colors")
    relationship_colors = appearance.get("relationship_colors") or snapshot.get("relationship_colors")
    graph_background = appearance.get("graph_background") or snapshot.get("graph_background")
    st.session_state.type_icons = appearance.get("type_icons", st.session_state.type_icons)
    community_colors = appearance.get("community_colors") or snapshot.get("community_colors")
    community_labels = appearance.get("community_labels") or snapshot.get("community_labels")
    community_locks = appearance.get("community_locks") or snapshot.get("community_locks")
    community_lock_labels = appearance.get("community_lock_labels") or snapshot.get("community_lock_labels")
    st.session_state.node_type_colors = _coerce_palette(
        node_type_colors, CONFIG["NODE_TYPE_COLORS"]
    )
    st.session_state.relationship_colors = _coerce_palette(
        relationship_colors, CONFIG["RELATIONSHIP_CONFIG"]
    )
    st.session_state.graph_background = _coerce_background(
        graph_background, CONFIG.get("GRAPH_BACKGROUND", {})
    )
    st.session_state.community_colors = _coerce_color_list(
        community_colors, CONFIG.get("COMMUNITY_COLORS", [])
    )
    default_labels = [f"Community {idx + 1}" for idx in range(len(st.session_state.community_colors))]
    st.session_state.community_labels = _coerce_string_list(community_labels, default_labels)
    st.session_state.community_locks = _coerce_lock_map(community_locks)
    st.session_state.community_lock_labels = _coerce_label_map(community_lock_labels)
    for node_type, color in st.session_state.node_type_colors.items():
        st.session_state[_palette_key("node_color", node_type)] = color
    for rel, color in st.session_state.relationship_colors.items():
        st.session_state[_palette_key("rel_color", rel)] = color
    if st.session_state.graph_background:
        st.session_state["graph_bg_radial_1"] = st.session_state.graph_background.get("radial_1")
        st.session_state["graph_bg_radial_2"] = st.session_state.graph_background.get("radial_2")
        st.session_state["graph_bg_linear_1"] = st.session_state.graph_background.get("linear_1")
        st.session_state["graph_bg_linear_2"] = st.session_state.graph_background.get("linear_2")
    for idx, color in enumerate(st.session_state.community_colors):
        st.session_state[f"community_color_{idx}"] = color
    for idx, label in enumerate(st.session_state.community_labels):
        st.session_state[f"community_label_{idx}"] = label
    for idx, lock in st.session_state.community_locks.items():
        st.session_state[f"community_lock_{idx}"] = lock
    for node_id, label in st.session_state.community_lock_labels.items():
        st.session_state[f"community_lock_label_{node_id}"] = label

    meta = snapshot.get("meta") or {}
    st.session_state.export_title = meta.get("export_title", "Linked Data Explorer")
    st.session_state.export_title_draft = st.session_state.export_title

    analytics = snapshot.get("analytics") or {}
    centrality_measures = analytics.get("centrality_measures")
    st.session_state.centrality_measures = (
        centrality_measures if isinstance(centrality_measures, dict) else None
    )
    graph_embeddings = analytics.get("graph_embeddings")
    st.session_state.graph_embeddings = graph_embeddings if isinstance(graph_embeddings, dict) else None
    textual_embeddings = analytics.get("textual_semantic_embeddings")
    st.session_state.textual_semantic_embeddings = (
        textual_embeddings if isinstance(textual_embeddings, dict) else None
    )

    try:
        st.session_state.rdf_graph = convert_graph_data_to_rdf(st.session_state.graph_data)
    except Exception as exc:
        st.session_state.rdf_graph = None
        st.error(f"Error rebuilding RDF graph from workspace: {exc}")


def _merge_graph_data(base: GraphData, incoming: GraphData) -> GraphData:
    if not base or not base.nodes:
        return incoming
    if not incoming or not incoming.nodes:
        return base

    node_map = {node.id: node for node in base.nodes if node.id}
    for new_node in incoming.nodes:
        node_id = new_node.id
        if not node_id:
            continue
        existing = node_map.get(node_id)
        if existing is None:
            node_map[node_id] = new_node
            continue

        if (
            _label_is_unnamed(existing.label, existing.id)
            and new_node.label
            and not _label_is_unnamed(new_node.label, new_node.id)
        ):
            existing.label = new_node.label

        existing_types = list(existing.types or [])
        for node_type in new_node.types or []:
            if node_type not in existing_types:
                existing_types.append(node_type)
        existing.types = existing_types

        if isinstance(existing.metadata, dict) and isinstance(new_node.metadata, dict):
            for key, value in new_node.metadata.items():
                if key not in existing.metadata:
                    existing.metadata[key] = value
                    continue
                if (
                    key == "583"
                    and isinstance(existing.metadata.get(key), list)
                    and isinstance(value, list)
                ):
                    for item in value:
                        if item not in existing.metadata[key]:
                            existing.metadata[key].append(item)
        elif not existing.metadata and new_node.metadata:
            existing.metadata = new_node.metadata

        existing_edge_keys = {(e.source, e.target, e.relationship) for e in existing.edges}
        for edge in new_node.edges:
            key = (edge.source, edge.target, edge.relationship)
            if key not in existing_edge_keys:
                existing.edges.append(edge)
                existing_edge_keys.add(key)

    return GraphData(nodes=list(node_map.values()))


def _apply_label_updates(graph_data: GraphData, id_to_label: Dict[str, str]) -> None:
    if not graph_data or not id_to_label:
        return
    for node in graph_data.nodes:
        candidate = id_to_label.get(node.id)
        if candidate and _label_is_unnamed(node.label, node.id) and not _label_is_unnamed(candidate, node.id):
            node.label = candidate
        if node.label and (
            node.id not in id_to_label or _label_is_unnamed(id_to_label.get(node.id), node.id)
        ):
            if not _label_is_unnamed(node.label, node.id):
                id_to_label[node.id] = node.label


def render_sidebar() -> SidebarState:
    def _graph_signature(graph: GraphData) -> Tuple[int, int, int, int]:
        node_count = len(graph.nodes)
        edge_count = sum(len(n.edges) for n in graph.nodes)
        node_checksum = 0
        edge_checksum = 0
        for node in graph.nodes:
            if node.id:
                node_checksum ^= hash(node.id)
            for edge in node.edges:
                edge_checksum ^= hash((edge.source, edge.target, edge.relationship))
        return (node_count, edge_count, node_checksum, edge_checksum)

    with st.sidebar.expander("File Upload"):
        collapse_oclc_nodes = st.checkbox(
            "Collapse OCLC sameAs nodes",
            value=True,
            key="collapse_oclc_nodes",
            help="Hide OCLC entity nodes that have sameAs links to other entities and rewire edges.",
        )
        uploaded_files = st.sidebar.file_uploader(
            "Upload JSON/RDF/RIS/MARC Files",
            type=["json", "jsonld", "ttl", "rdf", "nt", "ris", "dat", "mrc"],
            accept_multiple_files=True,
            help="Select files describing entities and relationships",
        )
        if not uploaded_files and "upload_signature" in st.session_state:
            st.session_state.upload_signature = None
            st.session_state.upload_errors = []
            st.session_state.ris_import_stats = None
            st.session_state.marc_import_stats = None
            st.session_state.oclc_link_stats = None
            st.session_state.oclc_collapse_stats = None
            st.session_state.oclc_base_graph = None
            st.session_state.oclc_base_labels = None
            st.session_state.oclc_base_signature = None
            st.session_state.oclc_collapsed_graph = None
            st.session_state.oclc_collapsed_labels = None
            st.session_state.oclc_collapsed_signature = None
            st.session_state.collapse_oclc_last = None
        if uploaded_files:
            upload_signature = [(file.name, file.size, file.type) for file in uploaded_files]
            if st.session_state.get("upload_signature") != upload_signature:
                st.session_state.upload_signature = upload_signature
                file_contents = []
                ris_contents = []
                marc_contents = []
                all_errors = []
                ris_errors = []
                ris_stats_total = {
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
                marc_stats_total = {
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
                for file in uploaded_files:
                    ext = file.name.split(".")[-1].lower()
                    if ext in ["ttl", "rdf", "nt"]:
                        graphs, errors = process_uploaded_file(file)
                        if graphs:
                            for g in graphs:
                                file_contents.append(g.serialize(format="json-ld").decode("utf-8"))
                        if errors:
                            all_errors.extend(errors)
                    elif ext == "ris":
                        try:
                            ris_contents.append(file.read().decode("utf-8"))
                        except Exception as exc:
                            all_errors.append(f"Error reading RIS file {file.name}: {exc}")
                    elif ext == "dat" or ext == "mrc" or ext == "mrk":
                        try:
                            marc_contents.append(file.read())
                        except Exception as exc:
                            all_errors.append(f"Error reading MARC file {file.name}: {exc}")
                    elif ext == "json" or ext == "jsonld" or ext == "json-ld":
                        try:
                            content = file.read().decode("utf-8")
                            # Validate that it's valid JSON
                            json.loads(content)
                            file_contents.append(content)
                        except json.JSONDecodeError as exc:
                            all_errors.append(f"Invalid JSON in file {file.name}: {exc}")
                        except Exception as exc:
                            all_errors.append(f"Error reading JSON file {file.name}: {exc}")
                    else:
                        try:
                            content = file.read().decode("utf-8")
                            # Try to parse as JSON; if it fails, skip it
                            try:
                                json.loads(content)
                                file_contents.append(content)
                            except json.JSONDecodeError:
                                all_errors.append(f"File {file.name} is not valid JSON (extension: .{ext}), skipping")
                        except Exception as exc:
                            all_errors.append(f"Error reading file {file.name}: {exc}")

                graph_data, id_to_label, errors = parse_entities_from_contents(file_contents)
                base_graph = st.session_state.graph_data if st.session_state.graph_data.nodes else GraphData(nodes=[])
                base_labels = dict(st.session_state.id_to_label or {})
                if graph_data.nodes:
                    base_graph = _merge_graph_data(base_graph, graph_data) if base_graph.nodes else graph_data
                    for node_id, label in id_to_label.items():
                        if node_id not in base_labels or _label_is_unnamed(base_labels.get(node_id), node_id):
                            if label and not _label_is_unnamed(label, node_id):
                                base_labels[node_id] = label

                if ris_contents:
                    for content in ris_contents:
                        ris_graph, ris_labels, ris_errs, ris_stats = parse_ris_content(
                            content, base_graph, base_labels
                        )
                        if ris_graph.nodes:
                            graph_data = (
                                _merge_graph_data(graph_data, ris_graph)
                                if graph_data.nodes
                                else ris_graph
                            )
                            base_graph = (
                                _merge_graph_data(base_graph, ris_graph)
                                if base_graph.nodes
                                else ris_graph
                            )
                            for node_id, label in ris_labels.items():
                                if node_id not in id_to_label or _label_is_unnamed(id_to_label.get(node_id), node_id):
                                    if label and not _label_is_unnamed(label, node_id):
                                        id_to_label[node_id] = label
                                if node_id not in base_labels or _label_is_unnamed(base_labels.get(node_id), node_id):
                                    if label and not _label_is_unnamed(label, node_id):
                                        base_labels[node_id] = label
                        if ris_errs:
                            ris_errors.extend(ris_errs)
                        if ris_stats:
                            for key, value in ris_stats.items():
                                ris_stats_total[key] = ris_stats_total.get(key, 0) + int(value)

                if marc_contents:
                    for content in marc_contents:
                        marc_graph, marc_labels, marc_errs, marc_stats = parse_marc_content(
                            content, base_graph, base_labels
                        )
                        if marc_graph.nodes:
                            graph_data = (
                                _merge_graph_data(graph_data, marc_graph)
                                if graph_data.nodes
                                else marc_graph
                            )
                            base_graph = (
                                _merge_graph_data(base_graph, marc_graph)
                                if base_graph.nodes
                                else marc_graph
                            )
                            for node_id, label in marc_labels.items():
                                if node_id not in id_to_label or _label_is_unnamed(
                                    id_to_label.get(node_id), node_id
                                ):
                                    if label and not _label_is_unnamed(label, node_id):
                                        id_to_label[node_id] = label
                                if node_id not in base_labels or _label_is_unnamed(
                                    base_labels.get(node_id), node_id
                                ):
                                    if label and not _label_is_unnamed(label, node_id):
                                        base_labels[node_id] = label
                        if marc_errs:
                            all_errors.extend(marc_errs)
                        if marc_stats:
                            for key, value in marc_stats.items():
                                marc_stats_total[key] = marc_stats_total.get(key, 0) + int(value)

                # (MARC parse errors are added to upload errors and shown in Data Validation)

                if st.session_state.graph_data.nodes:
                    graph_data = _merge_graph_data(st.session_state.graph_data, graph_data)
                    merged_labels = dict(st.session_state.id_to_label or {})
                    for node_id, label in id_to_label.items():
                        if node_id not in merged_labels or _label_is_unnamed(merged_labels.get(node_id), node_id):
                            if label and not _label_is_unnamed(label, node_id):
                                merged_labels[node_id] = label
                    id_to_label = merged_labels
                graph_data, id_to_label, oclc_stats = link_oclc_creators(graph_data, id_to_label)
                _apply_label_updates(graph_data, id_to_label)

                base_graph = graph_data
                base_labels = dict(id_to_label)
                base_signature = _graph_signature(base_graph)
                if collapse_oclc_nodes:
                    collapsed_graph, collapsed_labels, collapse_stats = collapse_same_as_nodes(
                        base_graph, base_labels
                    )
                    graph_data = collapsed_graph
                    id_to_label = collapsed_labels
                    collapsed_signature = _graph_signature(graph_data)
                else:
                    collapsed_graph = None
                    collapsed_labels = None
                    collapsed_signature = None
                    collapse_stats = None
                st.session_state.graph_data = graph_data
                st.session_state.id_to_label = id_to_label
                st.session_state.graph_embeddings = None
                st.session_state.textual_semantic_embeddings = None
                st.session_state.node_roles = None
                st.session_state.role_signature = None
                st.session_state.node_anomalies = None
                st.session_state.anomaly_signature = None
                st.session_state.ris_import_stats = ris_stats_total if ris_contents else None
                st.session_state.marc_import_stats = marc_stats_total if marc_contents else None
                st.session_state.oclc_link_stats = (
                    oclc_stats if oclc_stats and oclc_stats.get("targets", 0) > 0 else None
                )
                st.session_state.oclc_collapse_stats = (
                    collapse_stats
                    if collapse_stats and collapse_stats.get("aliases", 0) > 0
                    else None
                )
                st.session_state.oclc_base_graph = base_graph
                st.session_state.oclc_base_labels = base_labels
                st.session_state.oclc_base_signature = base_signature
                st.session_state.oclc_collapsed_graph = collapsed_graph
                st.session_state.oclc_collapsed_labels = collapsed_labels
                st.session_state.oclc_collapsed_signature = collapsed_signature
                st.session_state.collapse_oclc_last = collapse_oclc_nodes

                try:
                    st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
                except Exception as exc:
                    st.error(f"Error converting graph data to RDF: {exc}")

                combined_errors = []
                if errors:
                    combined_errors.extend(errors)
                if all_errors:
                    combined_errors.extend(all_errors)
                if ris_errors:
                    combined_errors.extend(ris_errors)
                st.session_state.upload_errors = combined_errors
            else:
                if st.session_state.get("oclc_base_graph") is not None:
                    last_state = st.session_state.get("collapse_oclc_last")
                    if last_state is None:
                        st.session_state.collapse_oclc_last = collapse_oclc_nodes
                    elif last_state != collapse_oclc_nodes:
                        if collapse_oclc_nodes:
                            collapsed_graph = st.session_state.get("oclc_collapsed_graph")
                            collapsed_labels = st.session_state.get("oclc_collapsed_labels")
                            collapse_stats = st.session_state.get("oclc_collapse_stats")
                            if collapsed_graph is None:
                                collapsed_graph, collapsed_labels, collapse_stats = collapse_same_as_nodes(
                                    st.session_state.oclc_base_graph,
                                    st.session_state.oclc_base_labels or {},
                                )
                                st.session_state.oclc_collapsed_graph = collapsed_graph
                                st.session_state.oclc_collapsed_labels = collapsed_labels
                                st.session_state.oclc_collapsed_signature = _graph_signature(collapsed_graph)
                                st.session_state.oclc_collapse_stats = (
                                    collapse_stats
                                    if collapse_stats and collapse_stats.get("aliases", 0) > 0
                                    else None
                                )
                            _apply_label_updates(collapsed_graph, collapsed_labels)
                            st.session_state.graph_data = collapsed_graph
                            st.session_state.id_to_label = collapsed_labels
                        else:
                            _apply_label_updates(
                                st.session_state.oclc_base_graph,
                                st.session_state.oclc_base_labels or {},
                            )
                        st.session_state.graph_data = st.session_state.oclc_base_graph
                        st.session_state.id_to_label = st.session_state.oclc_base_labels
                    st.session_state.graph_embeddings = None
                    st.session_state.textual_semantic_embeddings = None
                    st.session_state.node_roles = None
                    st.session_state.role_signature = None
                    st.session_state.node_anomalies = None
                    st.session_state.anomaly_signature = None
                    st.session_state.collapse_oclc_last = collapse_oclc_nodes
                    try:
                        st.session_state.rdf_graph = convert_graph_data_to_rdf(
                            st.session_state.graph_data
                        )
                    except Exception as exc:
                        st.error(f"Error converting graph data to RDF: {exc}")

        ris_stats = st.session_state.get("ris_import_stats")
        if ris_stats and ris_stats.get("records", 0) > 0:
            st.markdown("**RIS Import Summary**")
            st.caption(
                "Records: {records} | Works: {works_created} (direct {works_direct_matched}, "
                "merge {works_fuzzy_merged}, link {works_fuzzy_linked}, ambiguous {works_ambiguous})".format(
                    **ris_stats
                )
            )
            st.caption(
                "People: {persons_created} (merge {persons_merged}, link {persons_linked}) | "
                "Orgs: {orgs_created} (merge {orgs_merged}, link {orgs_linked}) | "
                "Edges: {edges_created}".format(**ris_stats)
            )
        marc_stats = st.session_state.get("marc_import_stats")
        if marc_stats and marc_stats.get("records", 0) > 0:
            st.markdown("**MARC Import Summary**")
            st.caption(
                "Records: {records} | Works: {works_created} (merge {works_fuzzy_merged}, "
                "link {works_fuzzy_linked}, ambiguous {works_ambiguous})".format(**marc_stats)
            )
            st.caption(
                "Actions: {actions_created} | People: {persons_created} (merge {persons_merged}, "
                "link {persons_linked}) | Orgs: {orgs_created} (merge {orgs_merged}, link {orgs_linked}) | "
                "Edges: {edges_created}".format(**marc_stats)
            )
        oclc_stats = st.session_state.get("oclc_link_stats")
        if oclc_stats and oclc_stats.get("targets", 0) > 0:
            st.markdown("**OCLC Link Summary**")
            st.caption(
                "OCLC creator IDs: {targets} | Resolved: {resolved} | Remapped edges: {mapped}".format(
                    **oclc_stats
                )
            )
            st.caption(
                "Remote lookups: {fetched} (failed {fetch_failed})".format(**oclc_stats)
            )
            if oclc_stats.get("resolved", 0) == 0:
                st.caption("No OCLC creator links resolved. Try re-uploading or check network access.")
        collapse_stats = st.session_state.get("oclc_collapse_stats")
        if collapse_stats and collapse_stats.get("aliases", 0) > 0:
            st.markdown("**OCLC Collapse Summary**")
            st.caption(
                "Hidden OCLC nodes: {aliases} | Rewired edges: {rewired}".format(**collapse_stats)
            )

        st.markdown("---")
        st.subheader("Remote SPARQL Endpoint")
        endpoint_url = st.text_input("Enter SPARQL Endpoint URL", key="sparql_endpoint_url")
        if st.button("Load from SPARQL Endpoint"):
            remote_graph, remote_id_to_label, sparql_errors = load_data_from_sparql(endpoint_url)
            if sparql_errors:
                for err in sparql_errors:
                    st.error(err)
            else:
                st.success("Data loaded from SPARQL endpoint.")
                st.session_state.graph_data = remote_graph
                st.session_state.id_to_label = remote_id_to_label
                st.session_state.graph_embeddings = None
                st.session_state.textual_semantic_embeddings = None
                st.session_state.node_roles = None
                st.session_state.role_signature = None
                st.session_state.node_anomalies = None
                st.session_state.anomaly_signature = None
                try:
                    st.session_state.rdf_graph = convert_graph_data_to_rdf(remote_graph)
                except Exception as exc:
                    st.error(f"Error converting SPARQL data to RDF: {exc}")

        st.markdown("---")
        st.subheader("SHACL Validation")
        if st.sidebar.checkbox("Upload SHACL Shapes for Validation"):
            shacl_file = st.sidebar.file_uploader("Upload SHACL file", type=["ttl", "turtle", "rdf"])
            if shacl_file and "rdf_graph" in st.session_state:
                shacl_content = shacl_file.read().decode("utf-8")
                conforms, report = validate_with_shacl(st.session_state.rdf_graph, shacl_content)
                if not conforms:
                    st.error("SHACL Validation Failed:")
                    st.text(report)
                else:
                    st.success("SHACL Validation Passed!")

        st.markdown("---")
        st.subheader("URI Dereferencing")
        if st.sidebar.checkbox("Enable URI Dereferencing"):
            uri_input = st.sidebar.text_area("Enter URIs (one per line)")
            if uri_input:
                uris = [line.strip() for line in uri_input.splitlines() if line.strip()]
                deref_graph = RDFGraph()
                messages = []
                for uri in uris:
                    result = dereference_uri(uri)
                    if result:
                        g, count = result
                        deref_graph += g
                        messages.append(f"URI '{uri}' fetched {count} triple(s).")

                if deref_graph:
                    if "rdf_graph" in st.session_state:
                        st.session_state.rdf_graph += deref_graph
                    else:
                        st.session_state.rdf_graph = deref_graph
                    st.success("Dereferenced URIs added to the RDF graph!")
                    for msg in messages:
                        st.info(msg)
                    triples = set(deref_graph)
                    if triples:
                        if "graph_data" not in st.session_state:
                            st.session_state.graph_data = GraphData(nodes=[])
                        if "id_to_label" not in st.session_state:
                            st.session_state.id_to_label = {}
                        st.session_state.graph_data, st.session_state.id_to_label, enhancement_stats = (
                            enhance_graph_data_from_triples(
                                st.session_state.graph_data,
                                st.session_state.id_to_label,
                                triples,
                            )
                        )
                        _apply_label_updates(
                            st.session_state.graph_data, st.session_state.id_to_label
                        )
                        st.session_state.graph_embeddings = None
                        st.session_state.textual_semantic_embeddings = None
                        st.session_state.centrality_measures = None
                        st.session_state.node_roles = None
                        st.session_state.role_signature = None
                        st.session_state.node_anomalies = None
                        st.session_state.anomaly_signature = None
                        if enhancement_stats and enhancement_stats.get("labels", 0) > 0:
                            st.info(
                                f"Updated {enhancement_stats['labels']} label(s) from dereferenced data."
                            )

    if st.session_state.upload_errors:
        with st.sidebar.expander("Data Validation"):
            for error in st.session_state.upload_errors:
                st.error(error)

    with st.sidebar.expander("Workspace"):
        st.caption("Save your graph, filters, annotations, and view settings to a file.")
        if st.session_state.get("workspace_name_pending"):
            st.session_state.workspace_name = st.session_state.workspace_name_pending
            st.session_state.workspace_name_input = st.session_state.workspace_name_pending
            st.session_state.workspace_name_pending = None
        st.text_input(
            "Workspace name",
            key="workspace_name_input",
            help="Used to label the saved workspace file.",
        )
        st.session_state.workspace_name = st.session_state.workspace_name_input
        workspace_payload = _build_workspace_snapshot()
        workspace_json = json.dumps(workspace_payload, ensure_ascii=True, indent=2, default=str)
        workspace_slug = slugify_filename(st.session_state.workspace_name or "workspace")
        st.download_button(
            "Save Workspace",
            data=workspace_json,
            file_name=f"{workspace_slug}.json",
            mime="application/json",
        )
        st.markdown("---")
        workspace_file = st.file_uploader("Load Workspace", type=["json"], key="workspace_file")
        if workspace_file is not None:
            if st.button("Load Workspace"):
                try:
                    snapshot = json.loads(workspace_file.getvalue().decode("utf-8"))
                    _restore_workspace(snapshot)
                    st.success("Workspace loaded.")
                except Exception as exc:
                    st.error(f"Failed to load workspace: {exc}")

        st.markdown("---")
        st.markdown("#### DigitalOcean Spaces")
        if not boto3_installed:
            st.info("Install boto3 to enable DigitalOcean Spaces workspace storage.")
        else:
            with st.form("digitalocean_spaces_form", clear_on_submit=False):
                default_access = _read_secret("DO_SPACES_KEY")
                default_secret = _read_secret("DO_SPACES_SECRET")
                default_region = _read_secret("DO_SPACES_REGION")
                default_bucket = _read_secret("DO_SPACES_BUCKET")
                default_prefix = _read_secret("DO_SPACES_PREFIX")
                default_endpoint = _read_secret("DO_SPACES_ENDPOINT")

                access_key = st.text_input(
                    "Spaces access key",
                    value=default_access,
                    type="password",
                    key="spaces_access_key",
                )
                secret_key = st.text_input(
                    "Spaces secret key",
                    value=default_secret,
                    type="password",
                    key="spaces_secret_key",
                )
                region = st.text_input(
                    "Region (e.g., nyc3)",
                    value=default_region,
                    key="spaces_region",
                )
                bucket = st.text_input(
                    "Space name",
                    value=default_bucket,
                    key="spaces_bucket",
                )
                prefix = st.text_input(
                    "Folder prefix (optional)",
                    value=default_prefix,
                    key="spaces_prefix",
                    help="Optional folder path inside the Space.",
                )
                endpoint_override = st.text_input(
                    "Endpoint URL (optional)",
                    value=default_endpoint,
                    key="spaces_endpoint",
                    help="Defaults to https://{region}.digitaloceanspaces.com.",
                )
                endpoint = _spaces_endpoint(region, endpoint_override)
                normalized_prefix = _normalize_spaces_prefix(prefix)
                spaces_ready = all(
                    value.strip()
                    for value in (
                        access_key,
                        secret_key,
                        region,
                        bucket,
                        endpoint,
                    )
                )

                save_to_spaces = st.form_submit_button(
                    "Save Workspace to Spaces"
                )
                if save_to_spaces:
                    if not spaces_ready:
                        st.warning("Please fill in all DigitalOcean Spaces fields first.")
                    else:
                        try:
                            client = _spaces_client(access_key, secret_key, region, endpoint)
                            remote_key = f"{normalized_prefix}{workspace_slug}.json"
                            _spaces_upload_workspace(client, bucket, remote_key, workspace_json)
                            st.success(f"Workspace saved to {remote_key}.")
                            st.session_state.spaces_workspace_list = None  # Clear cache
                        except (ClientError, BotoCoreError) as exc:
                            st.error(f"Spaces upload failed: {exc}")
                        except Exception as exc:
                            st.error(f"Unexpected error uploading workspace: {exc}")

                test_connection = st.form_submit_button("Test Connection")
                if test_connection:
                    if not spaces_ready:
                        st.warning("Please fill in all DigitalOcean Spaces fields first.")
                    else:
                        try:
                            client = _spaces_client(access_key, secret_key, region, endpoint)
                            # Try to list objects (max-keys 1) to verify access
                            client.list_objects_v2(Bucket=bucket, MaxKeys=1)
                            st.success(f"Successfully connected to Space: {bucket}")
                        except (ClientError, BotoCoreError) as exc:
                            st.error(f"Connection failed: {exc}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")

                refresh_spaces = st.form_submit_button(
                    "Refresh Spaces list"
                )
                if refresh_spaces:
                    if not spaces_ready:
                        st.warning("Please fill in all DigitalOcean Spaces fields first.")
                    else:
                        try:
                            client = _spaces_client(access_key, secret_key, region, endpoint)
                            st.session_state.spaces_workspace_list = _spaces_list_workspaces(
                                client, bucket, normalized_prefix
                            )
                            st.success("Spaces list refreshed!")
                        except (ClientError, BotoCoreError) as exc:
                            st.error(f"Spaces list failed: {exc}")
                        except Exception as exc:
                            st.error(f"Unexpected error listing workspaces: {exc}")

                remote_items = st.session_state.get("spaces_workspace_list") or []
                if remote_items:
                    options = [item["key"] for item in remote_items if item.get("key")]
                    selected_remote = st.selectbox(
                        "Remote workspaces",
                        options=options,
                        key="spaces_workspace_select",
                    )
                    load_remote = st.form_submit_button(
                        "Load Workspace from Spaces"
                    )
                    if load_remote:
                        if not spaces_ready:
                            st.warning("Please fill in all DigitalOcean Spaces fields first.")
                        else:
                            try:
                                client = _spaces_client(access_key, secret_key, region, endpoint)
                                payload = _spaces_download_workspace(client, bucket, selected_remote)
                                snapshot = json.loads(payload)
                                _restore_workspace(snapshot)
                                st.success("Workspace loaded from Spaces.")
                            except (ClientError, BotoCoreError) as exc:
                                st.error(f"Spaces download failed: {exc}")
                            except Exception as exc:
                                st.error(f"Failed to load workspace: {exc}")
                else:
                    st.caption("No remote workspaces loaded yet. Click 'Refresh Spaces list'.")

            st.markdown("---")
            with st.form("cdn_url_form"):
                cdn_url = st.text_input(
                    "Load workspace from URL (CDN/public)",
                    key="spaces_workspace_url",
                    placeholder="https://example.cdn.digitaloceanspaces.com/workspaces/my-workspace.json",
                )
                load_from_url = st.form_submit_button("Load Workspace from URL")
                if load_from_url:
                    try:
                        with st.spinner("Downloading workspace..."):
                            resp = requests.get(cdn_url.strip(), timeout=20)
                            resp.raise_for_status()
                            snapshot = resp.json()
                        _restore_workspace(snapshot)
                        st.success("Workspace loaded from URL.")
                    except Exception as exc:
                        st.error(f"Failed to load workspace from URL: {exc}")

    with st.sidebar.expander("Semantic Enhancements"):
        if "rdfs_last_signature" not in st.session_state:
            st.session_state.rdfs_last_signature = None
        if st.sidebar.button("Run RDFS Reasoning"):
            if "rdf_graph" not in st.session_state:
                st.info("Load data first to run RDFS reasoning.")
            elif not owlrl_installed:
                st.warning("owlrl is not installed. RDFS reasoning is unavailable.")
            else:
                signature = compute_rdfs_signature(st.session_state.graph_data, st.session_state.rdf_graph)
                if st.session_state.rdfs_last_signature == signature:
                    st.info("RDFS reasoning is already up to date for the current graph.")
                else:
                    new_graph, new_count, new_triples = apply_rdfs_reasoning(
                        st.session_state.rdf_graph, st.session_state.graph_data
                    )
                    st.session_state.rdf_graph = new_graph
                    enhancement_stats = None
                    if new_triples:
                        st.session_state.graph_data, st.session_state.id_to_label, enhancement_stats = (
                            enhance_graph_data_from_triples(
                                st.session_state.graph_data,
                                st.session_state.id_to_label,
                                new_triples,
                            )
                        )
                        st.session_state.graph_embeddings = None
                        st.session_state.textual_semantic_embeddings = None
                        st.session_state.centrality_measures = None
                        st.session_state.node_roles = None
                        st.session_state.role_signature = None
                        st.session_state.node_anomalies = None
                        st.session_state.anomaly_signature = None
                    st.session_state.rdfs_last_signature = compute_rdfs_signature(
                        st.session_state.graph_data, st.session_state.rdf_graph
                    )
                    st.success(f"RDFS reasoning applied! Added {new_count} new triple(s).")
                    if enhancement_stats and sum(enhancement_stats.values()) > 0:
                        st.info(
                            "Graph enhanced: "
                            f"+{enhancement_stats['nodes']} nodes, "
                            f"+{enhancement_stats['edges']} edges, "
                            f"+{enhancement_stats['types']} types, "
                            f"+{enhancement_stats['inferred_types']} inferred types, "
                            f"+{enhancement_stats['labels']} labels, "
                            f"+{enhancement_stats['properties']} properties."
                        )

        if st.sidebar.checkbox("Suggest Ontologies"):
            if "rdf_graph" in st.session_state:
                suggestions = suggest_ontologies(st.session_state.rdf_graph)
                if suggestions:
                    st.info("Suggested ontologies: " + ", ".join(suggestions))
                else:
                    st.info("No ontology suggestions available.")

    with st.sidebar.expander("Visualization Settings"):
        st.checkbox(
            "Performance mode (faster, fewer visuals)",
            key="performance_mode",
            help="Disables physics and animations, hides labels, and lowers motion intensity.",
        )
        if st.session_state.performance_mode and not st.session_state.performance_mode_applied:
            st.session_state.performance_backup = {
                "show_labels": st.session_state.show_labels,
                "smart_labels": st.session_state.smart_labels,
                "label_zoom_threshold": st.session_state.label_zoom_threshold,
                "reduce_motion": st.session_state.reduce_motion,
                "motion_intensity": st.session_state.motion_intensity,
                "node_animations": list(st.session_state.node_animations or []),
                "node_animation_strength": st.session_state.node_animation_strength,
                "enable_physics": st.session_state.enable_physics,
                "physics_preset": st.session_state.get("physics_preset", "Default (Balanced)"),
                "physics_params": st.session_state.physics_params.copy(),
                "focus_context": st.session_state.focus_context,
                "edge_semantics": st.session_state.edge_semantics,
                "type_icons": st.session_state.type_icons,
            }
            st.session_state.show_labels = False
            st.session_state.smart_labels = False
            st.session_state.reduce_motion = True
            st.session_state.motion_intensity = 10
            st.session_state.node_animations = ["none"]
            st.session_state.node_animation_strength = 0
            st.session_state.enable_physics = False
            st.session_state.physics_preset = "No Physics (Manual Layout)"
            st.session_state.physics_params = {
                "gravity": 0,
                "centralGravity": 0,
                "springLength": 150,
                "springStrength": 0,
            }
            st.session_state.focus_context = False
            st.session_state.edge_semantics = False
            st.session_state.type_icons = False
            st.session_state.performance_mode_applied = True
        elif not st.session_state.performance_mode and st.session_state.performance_mode_applied:
            backup = st.session_state.performance_backup or {}
            st.session_state.show_labels = backup.get("show_labels", st.session_state.show_labels)
            st.session_state.smart_labels = backup.get("smart_labels", st.session_state.smart_labels)
            st.session_state.label_zoom_threshold = backup.get(
                "label_zoom_threshold", st.session_state.label_zoom_threshold
            )
            st.session_state.reduce_motion = backup.get("reduce_motion", st.session_state.reduce_motion)
            st.session_state.motion_intensity = backup.get("motion_intensity", st.session_state.motion_intensity)
            st.session_state.node_animations = backup.get("node_animations", st.session_state.node_animations)
            st.session_state.node_animation_strength = backup.get(
                "node_animation_strength", st.session_state.node_animation_strength
            )
            st.session_state.enable_physics = backup.get("enable_physics", st.session_state.enable_physics)
            st.session_state.physics_preset = backup.get(
                "physics_preset", st.session_state.get("physics_preset", "Default (Balanced)")
            )
            st.session_state.physics_params = backup.get("physics_params", st.session_state.physics_params)
            st.session_state.focus_context = backup.get("focus_context", st.session_state.focus_context)
            st.session_state.edge_semantics = backup.get("edge_semantics", st.session_state.edge_semantics)
            st.session_state.type_icons = backup.get("type_icons", st.session_state.type_icons)
            st.session_state.performance_mode_applied = False

        st.checkbox("Show Node Labels", key="show_labels")
        st.checkbox(
            "Smart labels (hover + zoom)",
            key="smart_labels",
            help="Hide labels until hover or zoom to reduce clutter on dense graphs.",
            disabled=not st.session_state.show_labels,
        )
        if st.session_state.show_labels and st.session_state.smart_labels:
            st.slider(
                "Label zoom threshold",
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                key="label_zoom_threshold",
                help="Higher values show labels only when zoomed in closer.",
            )
        st.text_input(
            "Search nodes",
            key="search_term",
            help="Highlights nodes whose labels or IDs match the query.",
        )
        if st.session_state.search_term.strip() and st.session_state.graph_data.nodes:
            st.session_state.search_nodes = find_search_nodes(
                st.session_state.graph_data, st.session_state.search_term
            )
            if st.session_state.search_nodes:
                st.caption(f"Search matches: {len(st.session_state.search_nodes)} node(s).")
            else:
                st.caption("No matches for this search.")
        else:
            st.session_state.search_nodes = []
        st.checkbox("Reduce motion (less animation)", key="reduce_motion")
        st.checkbox(
            "Focus + context (dim non-selected)",
            key="focus_context",
            help="When selecting nodes, dim everything else to keep context.",
        )
        st.checkbox(
            "Edge semantics (direction/weight/inferred)",
            key="edge_semantics",
            help="Use dashes for inferred links and width for confidence/weight when available.",
        )
        st.checkbox(
            "Type icons in graph",
            key="type_icons",
            help="Replace node shapes with type badges.",
        )
        st.slider(
            "Motion intensity",
            min_value=0,
            max_value=100,
            step=5,
            help="Lower values stabilize faster; 0 is almost still, 100 is full motion.",
            key="motion_intensity",
        )
        node_animation_labels = {
            "none": "None",
            "selected": "Pulse (selected nodes + edges)",
            "focus": "Focus (dim non-selected)",
            "neighbors": "Neighbor halo (1-hop)",
            "search": "Pulse (search results)",
            "search_ping": "Search ping (one-shot)",
            "path": "Shortest-path glow",
            "centrality": "Centrality breathe",
            "all": "Pulse (all nodes)",
            "flow": "Flowing edges (direction)",
        }
        node_animation_help = {
            "selected": "Click nodes to pulse selections and their edges.",
            "focus": "Click a node to pulse it while dimming everything else.",
            "neighbors": "Click a node to pulse its 1-hop neighbors.",
            "search": "Uses the search term to pulse matching nodes.",
            "search_ping": "Runs a single pulse across search results.",
            "path": "Pulses nodes and edges on the computed shortest path.",
            "centrality": "Scales pulse size by degree centrality.",
            "all": "Pulses all nodes in the graph.",
            "flow": "Animates dashed edges to show direction.",
        }
        st.multiselect(
            "Animation modes",
            options=list(node_animation_labels.keys()),
            format_func=lambda key: node_animation_labels[key],
            key="node_animations",
        )
        selected_modes = list(st.session_state.node_animations or [])
        if "none" in selected_modes and len(selected_modes) > 1:
            selected_modes = [mode for mode in selected_modes if mode != "none"]
            st.session_state.node_animations = selected_modes
        effective_node_animations = [mode for mode in selected_modes if mode != "none"]
        st.slider(
            "Animation strength",
            min_value=0,
            max_value=100,
            step=5,
            key="node_animation_strength",
            disabled=st.session_state.reduce_motion or not effective_node_animations,
        )
        if effective_node_animations:
            for mode in effective_node_animations:
                help_text = node_animation_help.get(mode, "")
                if help_text:
                    st.caption(f"{node_animation_labels.get(mode, mode)}: {help_text}")
        active_mode_set = set(effective_node_animations)
        if active_mode_set.intersection({"search", "search_ping"}) and not st.session_state.search_term.strip():
            st.caption("Enter a search term to target search animations.")
        if st.session_state.search_term.strip() and not active_mode_set.intersection({"search", "search_ping"}):
            st.caption("Switch to a search animation mode to see motion on matches.")
        if "path" in active_mode_set and not st.session_state.shortest_path:
            st.caption("Run pathfinding to animate the shortest path.")
        if "centrality" in active_mode_set and not st.session_state.centrality_measures:
            if st.session_state.graph_data.nodes:
                st.session_state.centrality_measures = compute_centrality_measures(
                    st.session_state.graph_data
                )
        community_detection = st.checkbox("Enable Community Detection", key="community_detection")

        physics_presets = {
            "Default (Balanced)": CONFIG["PHYSICS_DEFAULTS"],
            "High Gravity (Clustering)": {
                "gravity": -100,
                "centralGravity": 0.05,
                "springLength": 100,
                "springStrength": 0.15,
            },
            "No Physics (Manual Layout)": {
                "gravity": 0,
                "centralGravity": 0,
                "springLength": 150,
                "springStrength": 0,
            },
            "Custom": st.session_state.physics_params,
        }
        preset_name = st.selectbox(
            "Physics Presets", list(physics_presets.keys()), index=0, key="physics_preset"
        )

        if preset_name != "Custom":
            st.session_state.physics_params = physics_presets[preset_name]
            st.info("Physics parameters set to preset: " + preset_name)
        else:
            st.subheader("Advanced Physics Settings")
            st.session_state.physics_params["gravity"] = st.number_input(
                "Gravity",
                value=float(
                    st.session_state.physics_params.get(
                        "gravity", CONFIG["PHYSICS_DEFAULTS"]["gravity"]
                    )
                ),
                step=1.0,
                key="gravity_input",
            )
            st.session_state.physics_params["centralGravity"] = st.number_input(
                "Central Gravity",
                value=float(
                    st.session_state.physics_params.get(
                        "centralGravity", CONFIG["PHYSICS_DEFAULTS"]["centralGravity"]
                    )
                ),
                step=0.01,
                key="centralGravity_input",
            )
            st.session_state.physics_params["springLength"] = st.number_input(
                "Spring Length",
                value=float(
                    st.session_state.physics_params.get(
                        "springLength", CONFIG["PHYSICS_DEFAULTS"]["springLength"]
                    )
                ),
                step=1.0,
                key="springLength_input",
            )
            st.session_state.physics_params["springStrength"] = st.number_input(
                "Spring Strength",
                value=float(
                    st.session_state.physics_params.get(
                        "springStrength", CONFIG["PHYSICS_DEFAULTS"]["springStrength"]
                    )
                ),
                step=0.01,
                key="springStrength_input",
            )

        enable_centrality = st.checkbox("Display Centrality Measures", value=False, key="centrality_enabled")
        if enable_centrality and st.session_state.graph_data.nodes:
            st.session_state.centrality_measures = compute_centrality_measures(
                st.session_state.graph_data
            )
            st.info("Centrality measures computed.")

    with st.sidebar.expander("Appearance"):
        st.caption("Customize graph colors and background (saved with workspace files).")

        node_palette = _coerce_palette(
            st.session_state.get("node_type_colors"), CONFIG["NODE_TYPE_COLORS"]
        )
        rel_palette = _coerce_palette(
            st.session_state.get("relationship_colors"), CONFIG["RELATIONSHIP_CONFIG"]
        )
        bg_defaults = CONFIG.get(
            "GRAPH_BACKGROUND",
            {"radial_1": "#E07A5F", "radial_2": "#2A9D8F", "linear_1": "#FCFAF6", "linear_2": "#F4EFE6"},
        )
        bg_palette = _coerce_background(st.session_state.get("graph_background"), bg_defaults)

        st.subheader("Graph Background")
        bg_cols = st.columns(2)
        bg_radial_1 = bg_cols[0].color_picker(
            "Radial Accent 1", value=bg_palette["radial_1"], key="graph_bg_radial_1"
        )
        bg_radial_2 = bg_cols[1].color_picker(
            "Radial Accent 2", value=bg_palette["radial_2"], key="graph_bg_radial_2"
        )
        bg_linear_1 = bg_cols[0].color_picker(
            "Linear Start", value=bg_palette["linear_1"], key="graph_bg_linear_1"
        )
        bg_linear_2 = bg_cols[1].color_picker(
            "Linear End", value=bg_palette["linear_2"], key="graph_bg_linear_2"
        )
        st.session_state.graph_background = {
            "radial_1": bg_radial_1,
            "radial_2": bg_radial_2,
            "linear_1": bg_linear_1,
            "linear_2": bg_linear_2,
        }

        st.markdown("---")
        st.subheader("Node Type Colors")
        node_items = sorted(node_palette.items(), key=lambda item: item[0].lower())
        node_cols = st.columns(2)
        updated_node_palette: Dict[str, str] = {}
        for idx, (node_type, color) in enumerate(node_items):
            col = node_cols[idx % 2]
            picker_key = _palette_key("node_color", node_type)
            updated_node_palette[node_type] = col.color_picker(
                node_type, value=color, key=picker_key
            )
        st.session_state.node_type_colors = updated_node_palette

        st.markdown("---")
        show_relationship_colors = st.checkbox("Show Relationship Colors", value=False)
        if show_relationship_colors:
            rel_items = sorted(rel_palette.items(), key=lambda item: item[0].lower())
            rel_cols = st.columns(2)
            updated_rel_palette: Dict[str, str] = {}
            for idx, (rel, color) in enumerate(rel_items):
                col = rel_cols[idx % 2]
                picker_key = _palette_key("rel_color", rel)
                updated_rel_palette[rel] = col.color_picker(rel, value=color, key=picker_key)
            st.session_state.relationship_colors = updated_rel_palette

        st.markdown("---")
        show_community_colors = st.checkbox("Show Community Palette", value=False)
        if show_community_colors:
            community_palette = _coerce_color_list(
                st.session_state.get("community_colors"), CONFIG.get("COMMUNITY_COLORS", [])
            )
            default_labels = [f"Community {idx + 1}" for idx in range(len(community_palette))]
            community_labels = _coerce_string_list(
                st.session_state.get("community_labels"), default_labels
            )
            if len(community_labels) < len(community_palette):
                community_labels.extend(default_labels[len(community_labels) :])
            community_locks = _coerce_lock_map(st.session_state.get("community_locks"))
            community_lock_labels = _coerce_label_map(
                st.session_state.get("community_lock_labels")
            )
            # Capture current widget state so add/remove doesn't discard edits
            widget_palette: List[str] = []
            widget_labels: List[str] = []
            widget_locks: Dict[int, str] = {}
            for idx, color in enumerate(community_palette):
                picker_key = f"community_color_{idx}"
                label_key = f"community_label_{idx}"
                lock_key = f"community_lock_{idx}"
                widget_palette.append(st.session_state.get(picker_key, color))
                widget_labels.append(st.session_state.get(label_key, community_labels[idx]))
                lock_value = st.session_state.get(lock_key, community_locks.get(idx, ""))
                if lock_value:
                    widget_locks[idx] = lock_value
            community_palette = widget_palette
            community_labels = widget_labels
            community_locks = widget_locks
            lock_labels_snapshot = dict(community_lock_labels)
            for slot_idx, node_id in community_locks.items():
                if slot_idx < len(community_labels):
                    label_value = str(community_labels[slot_idx]).strip()
                    if label_value:
                        lock_labels_snapshot[node_id] = label_value

            add_col, remove_col = st.columns(2)
            if add_col.button("Add Community Slot"):
                new_idx = len(community_palette)
                community_palette.append(_generate_community_color(new_idx))
                community_labels.append(f"Community {new_idx + 1}")
                _sync_community_widget_state(community_palette, community_labels, community_locks)
                st.session_state.community_colors = community_palette
                st.session_state.community_labels = community_labels
                st.session_state.community_locks = community_locks
                st.session_state.community_lock_labels = lock_labels_snapshot
                st.rerun()
            if remove_col.button("Remove Last Slot", disabled=len(community_palette) <= 1):
                community_palette.pop()
                community_labels = community_labels[: len(community_palette)]
                community_locks.pop(len(community_palette), None)
                _sync_community_widget_state(community_palette, community_labels, community_locks)
                st.session_state.community_colors = community_palette
                st.session_state.community_labels = community_labels
                st.session_state.community_locks = community_locks
                st.session_state.community_lock_labels = lock_labels_snapshot
                st.rerun()

            updated_community_palette: List[str] = []
            updated_community_labels: List[str] = []
            updated_community_locks: Dict[int, str] = {}
            updated_lock_labels: Dict[str, str] = dict(community_lock_labels)
            lock_options = [""] + [n.id for n in st.session_state.graph_data.nodes]
            label_map = {
                n.id: st.session_state.id_to_label.get(n.id, n.label or n.id)
                for n in st.session_state.graph_data.nodes
            }
            def _lock_label(nid: str) -> str:
                if not nid:
                    return "None"
                return label_map.get(nid, _shorten_iri(nid))
            for idx, color in enumerate(community_palette):
                left, middle, right = st.columns([1, 2, 2])
                picker_key = f"community_color_{idx}"
                label_key = f"community_label_{idx}"
                lock_key = f"community_lock_{idx}"
                color_value = st.session_state.get(picker_key, color)
                updated_community_palette.append(
                    left.color_picker(f"Color {idx + 1}", value=color_value, key=picker_key)
                )
                current_lock = community_locks.get(idx, "")
                if current_lock not in lock_options:
                    current_lock = ""
                label_default = (
                    community_lock_labels.get(current_lock, "")
                    if current_lock
                    else community_labels[idx] if idx < len(community_labels) else f"Community {idx + 1}"
                )
                label_value = st.session_state.get(label_key, label_default)
                updated_community_labels.append(
                    middle.text_input(
                        f"Label {idx + 1}",
                        value=label_value,
                        key=label_key,
                    )
                )
                selected_lock = right.selectbox(
                    f"Lock {idx + 1}",
                    options=lock_options,
                    index=lock_options.index(current_lock),
                    format_func=_lock_label,
                    key=lock_key,
                )
                if selected_lock:
                    lock_label = updated_community_labels[-1].strip()
                    if lock_label:
                        updated_lock_labels[selected_lock] = lock_label
                if selected_lock:
                    updated_community_locks[idx] = selected_lock
            st.session_state.community_colors = updated_community_palette
            st.session_state.community_labels = updated_community_labels
            st.session_state.community_locks = updated_community_locks
            st.session_state.community_lock_labels = updated_lock_labels

    with st.sidebar.expander("Graph Embeddings"):
        if not node2vec_installed:
            st.error("node2vec is not installed. Install it to compute graph embeddings.")
        if st.button("Compute Graph Embeddings"):
            if not node2vec_installed:
                st.error("Cannot compute embeddings without node2vec installed.")
                st.session_state.graph_embeddings = None
            elif not st.session_state.graph_data.nodes:
                st.warning("Load graph data before computing embeddings.")
                st.session_state.graph_embeddings = None
            else:
                embeddings = compute_probabilistic_graph_embeddings(st.session_state.graph_data)
                if embeddings:
                    st.session_state.graph_embeddings = embeddings
                    st.success("Graph Embeddings computed!")
                else:
                    st.session_state.graph_embeddings = None
                    st.error("Graph embeddings failed to compute. Check logs for details.")

    with st.sidebar.expander("Graph Pathfinding"):
        if st.session_state.graph_data.nodes:
            node_options = {
                n.id: st.session_state.id_to_label.get(n.id, n.id)
                for n in st.session_state.graph_data.nodes
            }
            source_pf = st.selectbox(
                "Source Node",
                options=list(node_options.keys()),
                format_func=lambda x: node_options[x],
                key="pf_source",
            )
            target_pf = st.selectbox(
                "Target Node",
                options=list(node_options.keys()),
                format_func=lambda x: node_options[x],
                key="pf_target",
            )

            if st.button("Find Shortest Path"):
                try:
                    G_pf = nx.DiGraph()
                    for n in st.session_state.graph_data.nodes:
                        G_pf.add_node(n.id)
                    for n in st.session_state.graph_data.nodes:
                        for e in n.edges:
                            G_pf.add_edge(e.source, e.target)

                    sp = nx.shortest_path(G_pf, source=source_pf, target=target_pf)
                    st.session_state.shortest_path = sp
                    st.success(f"Shortest path found with {len(sp) - 1} edges.")
                except Exception as exc:
                    st.session_state.shortest_path = None
                    st.error(f"Pathfinding failed: {exc}")
        else:
            st.info("No nodes available for pathfinding.")

    with st.sidebar.expander("Graph Editing"):
        nodes_list = st.session_state.graph_data.nodes
        node_label_map = {
            n.id: st.session_state.id_to_label.get(n.id, n.label or n.id) for n in nodes_list
        }

        def _node_label(nid: str) -> str:
            label = node_label_map.get(nid)
            return label if label else _shorten_iri(nid)

        ge_mode = st.selectbox(
            "Editing Mode",
            ["Add Node", "Delete Node", "Modify Node", "Add Edge", "Delete Edge"],
            key="graph_edit_mode",
        )

        if ge_mode == "Add Node":
            with st.form("add_node_form"):
                new_label = st.text_input("Node Label")
                new_type = st.selectbox("Node Type", list(CONFIG["NODE_TYPE_COLORS"].keys()))
                if st.form_submit_button("Add Node"):
                    if new_label:
                        nid = f"node_{int(time.time())}"
                        new_node = Node(
                            id=nid,
                            label=new_label,
                            types=[new_type],
                            metadata={"id": nid, "prefLabel": {"en": new_label}, "type": [new_type]},
                        )
                        st.session_state.graph_data.nodes.append(new_node)
                        st.session_state.id_to_label[nid] = new_label
                        st.success(f"Node '{new_label}' added!")
                    else:
                        st.error("Please provide a Node Label.")

        elif ge_mode == "Delete Node":
            nid_list = [n.id for n in nodes_list]
            if nid_list:
                node_to_delete = st.selectbox("Select Node to Delete", nid_list, format_func=_node_label)
                if st.button("Delete Node"):
                    deleted_label = _node_label(node_to_delete)
                    st.session_state.graph_data.nodes = [n for n in nodes_list if n.id != node_to_delete]
                    for node in st.session_state.graph_data.nodes:
                        node.edges = [e for e in node.edges if e.target != node_to_delete]
                    st.session_state.id_to_label.pop(node_to_delete, None)
                    st.success(f"Node '{deleted_label}' deleted.")
            else:
                st.info("No nodes to delete.")

        elif ge_mode == "Modify Node":
            nid_list = [n.id for n in nodes_list]
            if nid_list:
                node_to_modify = st.selectbox("Select Node to Modify", nid_list, format_func=_node_label)
                node_obj = next((n for n in nodes_list if n.id == node_to_modify), None)
                if node_obj:
                    with st.form("modify_node_form"):
                        new_label = st.text_input("New Label", node_obj.label)
                        current_type = node_obj.types[0] if node_obj.types else "Unknown"
                        new_type_options = list(CONFIG["NODE_TYPE_COLORS"].keys())
                        new_type_index = (
                            new_type_options.index(current_type)
                            if current_type in new_type_options
                            else 0
                        )
                        new_type_choice = st.selectbox(
                            "New Type", new_type_options, index=new_type_index
                        )

                        if st.form_submit_button("Modify Node"):
                            node_obj.label = new_label
                            node_obj.types = [new_type_choice]
                            node_obj.metadata["prefLabel"]["en"] = new_label
                            st.session_state.id_to_label[node_to_modify] = new_label
                            st.success(f"Node '{new_label}' modified.")
            else:
                st.info("No nodes to modify.")

        elif ge_mode == "Add Edge":
            if nodes_list:
                with st.form("add_edge_form"):
                    src_node = st.selectbox(
                        "Source Node", [n.id for n in nodes_list], format_func=_node_label
                    )
                    tgt_node = st.selectbox(
                        "Target Node", [n.id for n in nodes_list], format_func=_node_label
                    )
                    rel = st.selectbox("Relationship", list(CONFIG["RELATIONSHIP_CONFIG"].keys()))
                    if st.form_submit_button("Add Edge"):
                        for n in nodes_list:
                            if n.id == src_node:
                                n.edges.append(Edge(source=src_node, target=tgt_node, relationship=rel))
                        st.success(
                            f"Edge '{rel}' from '{_node_label(src_node)}' to '{_node_label(tgt_node)}' added."
                        )
            else:
                st.info("No nodes to add edges to.")

        elif ge_mode == "Delete Edge":
            all_edges = []
            for n in nodes_list:
                for e in n.edges:
                    all_edges.append((e.source, e.target, e.relationship))

            if all_edges:
                edge_to_delete = st.selectbox(
                    "Select Edge to Delete",
                    all_edges,
                    format_func=lambda e: f"{_node_label(e[0])} -> {_node_label(e[1])} ({e[2]})",
                )
                if st.button("Delete Edge"):
                    for n in nodes_list:
                        if n.id == edge_to_delete[0]:
                            n.edges = [
                                e
                                for e in n.edges
                                if (e.source, e.target, e.relationship) != edge_to_delete
                            ]
                    st.success("Edge deleted.")
            else:
                st.info("No edges to delete.")

    with st.sidebar.expander("Manual Node Positioning"):
        if st.session_state.graph_data.nodes:
            unique_nodes = {n.id: n.label for n in st.session_state.graph_data.nodes}
            sel_node = st.selectbox(
                "Select a Node to Position",
                list(unique_nodes.keys()),
                format_func=lambda x: unique_nodes[x],
                key="selected_node_control",
            )
            st.session_state.selected_node = sel_node
            cur_pos = st.session_state.node_positions.get(sel_node, {"x": 0.0, "y": 0.0})

            with st.form("position_form"):
                x_val = st.number_input("X Position", value=cur_pos["x"], step=10.0)
                y_val = st.number_input("Y Position", value=cur_pos["y"], step=10.0)
                if st.form_submit_button("Set Position"):
                    st.session_state.node_positions[sel_node] = {"x": x_val, "y": y_val}
                    st.success(
                        f"Position for '{unique_nodes[sel_node]}' set to (X: {x_val}, Y: {y_val})"
                    )

    with st.sidebar.expander("Advanced Filtering"):
        st.subheader("Property-based Filtering")
        prop_name = st.text_input("Property Name", key="filter_prop_name")
        prop_value = st.text_input("Property Value", key="filter_prop_value")
        if st.button("Apply Property Filter"):
            st.session_state.property_filter = {"property": prop_name, "value": prop_value}
            st.success("Property filter applied.")

        all_rels = set()
        for n in st.session_state.graph_data.nodes:
            for e in n.edges:
                all_rels.add(e.relationship)

        chosen_rels = st.multiselect("Select Relationship Types", sorted(all_rels), default=list(all_rels))
        st.session_state.selected_relationships = (
            chosen_rels if chosen_rels else list(CONFIG["RELATIONSHIP_CONFIG"].keys())
        )

        st.subheader("Filter by Node Type")
        unique_types = sorted(
            {canonical_type(t) for n in st.session_state.graph_data.nodes for t in n.types}
        )
        chosen_types = st.multiselect(
            "Select Node Types",
            options=unique_types,
            default=unique_types,
            key="filter_node_types",
        )
        st.session_state.filtered_types = chosen_types

        st.subheader("Filter by Inferred Type")
        inferred_type_options = sorted(
            {t for n in st.session_state.graph_data.nodes for t in _extract_inferred_types(n.metadata)}
        )
        if inferred_type_options:
            filtered_defaults = [
                t for t in st.session_state.filtered_inferred_types if t in inferred_type_options
            ]
            st.session_state.filtered_inferred_types = filtered_defaults
            st.multiselect(
                "Select Inferred Types",
                options=inferred_type_options,
                default=filtered_defaults,
                format_func=_shorten_iri,
                key="filtered_inferred_types",
                help="Filter nodes that include any of the selected inferred types.",
            )
        else:
            st.caption("No inferred types available for filtering.")

        st.subheader("Filter by Node Role")
        role_options = {
            "hub": "Hub (top degree centrality)",
            "bridge": "Bridge (top betweenness)",
            "outlier": "Outlier (low degree & closeness)",
        }
        role_defaults = [role for role in st.session_state.role_filters if role in role_options]
        st.session_state.role_filters = role_defaults
        st.multiselect(
            "Select Roles",
            options=list(role_options.keys()),
            default=role_defaults,
            format_func=lambda key: role_options.get(key, key),
            key="role_filters",
            help=(
                "Classifies nodes using centrality percentiles "
                "(default: hubs/bridges >=85th, outliers <=15th)."
            ),
        )

        st.subheader("Filter by Anomaly Flags")
        anomaly_options = {
            "rare_relationship_pattern": "Rare relationship pattern",
            "conflicting_types": "Conflicting types",
            "low_property_coverage": "Low property coverage",
        }
        anomaly_defaults = [
            flag for flag in st.session_state.anomaly_filters if flag in anomaly_options
        ]
        st.session_state.anomaly_filters = anomaly_defaults
        st.multiselect(
            "Select Flags",
            options=list(anomaly_options.keys()),
            default=anomaly_defaults,
            format_func=lambda key: anomaly_options.get(key, key),
            key="anomaly_filters",
            help=(
                "Flags nodes with rare relationship patterns, conflicting types, "
                "or low metadata coverage (bottom percentiles)."
            ),
        )

    st.session_state.sparql_query = st.sidebar.text_area(
        "SPARQL Query",
        help="Enter a SPARQL SELECT query to filter nodes.",
        key="sparql_query_control",
        value=st.session_state.sparql_query,
    )

    if st.session_state.sparql_query.strip():
        st.sidebar.info("Query Running...")
        try:
            filtered_nodes = run_sparql_query(
                st.session_state.sparql_query, st.session_state.rdf_graph
            )
            st.sidebar.success(f"Query Successful: {len(filtered_nodes)} result(s) found.")
            st.sidebar.dataframe(
                pd.DataFrame(list(filtered_nodes), columns=["Node ID"]),
                use_container_width=True,
            )
        except Exception as exc:
            st.sidebar.error(f"SPARQL Query failed: {exc}")
            filtered_nodes = None
    else:
        filtered_nodes = None

    if st.session_state.filtered_types:
        filter_by_type = {
            n.id
            for n in st.session_state.graph_data.nodes
            if any(canonical_type(t) in st.session_state.filtered_types for t in n.types)
        }
        filtered_nodes = filtered_nodes.intersection(filter_by_type) if filtered_nodes is not None else filter_by_type

    if st.session_state.filtered_inferred_types:
        selected_inferred = set(st.session_state.filtered_inferred_types)
        filter_by_inferred = {
            n.id
            for n in st.session_state.graph_data.nodes
            if selected_inferred.intersection(set(_extract_inferred_types(n.metadata)))
        }
        filtered_nodes = (
            filtered_nodes.intersection(filter_by_inferred) if filtered_nodes is not None else filter_by_inferred
        )

    if st.session_state.role_filters:
        signature = _graph_signature(st.session_state.graph_data)
        if st.session_state.role_signature != signature:
            st.session_state.node_roles = compute_node_roles(st.session_state.graph_data)
            st.session_state.role_signature = signature
        selected_roles = set(st.session_state.role_filters or [])
        role_nodes = {
            node_id
            for node_id, roles in (st.session_state.node_roles or {}).items()
            if selected_roles.intersection(set(roles or []))
        }
        filtered_nodes = filtered_nodes.intersection(role_nodes) if filtered_nodes is not None else role_nodes

    if st.session_state.anomaly_filters:
        signature = _graph_signature(st.session_state.graph_data)
        if st.session_state.anomaly_signature != signature:
            st.session_state.node_anomalies = compute_anomaly_flags(st.session_state.graph_data)
            st.session_state.anomaly_signature = signature
        selected_flags = set(st.session_state.anomaly_filters or [])
        anomaly_nodes = {
            node_id
            for node_id, flags in (st.session_state.node_anomalies or {}).items()
            if selected_flags.intersection(set(flags or []))
        }
        filtered_nodes = (
            filtered_nodes.intersection(anomaly_nodes) if filtered_nodes is not None else anomaly_nodes
        )

    return SidebarState(
        filtered_nodes=filtered_nodes,
        effective_node_animations=effective_node_animations,
        community_detection=community_detection,
    )
