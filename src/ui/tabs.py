"""Logic for specific tabs (Graph, Data, etc.)."""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components
from dateutil.parser import parse as parse_date

from src.config import CONFIG, GRAPH_CARD_HEIGHT, MAX_RENDER_NODES, NAMESPACE_PREFIXES
from src.data_processing import (
    apply_render_cap,
    compute_textual_semantic_embeddings,
    convert_graph_data_to_rdf,
    get_edge_relationship,
    load_jsonld_into_session,
    oclc_sparql,
    run_sparql_query,
    sentence_transformer_installed,
)
from src.models import Edge, Node
from src.utils import (
    _blend_hex,
    _coerce_list,
    _extract_inferred_types,
    _format_conservation_action_block,
    _label_is_unnamed,
    _metadata_rows,
    _normalize_getty_ulan,
    _render_metadata_overview,
    _render_type_badges,
    _shorten_iri,
    _truncate_text,
    canonical_type,
    is_valid_iiif_manifest,
    remove_fragment,
)
from src.visualizer import (
    build_export_html,
    build_graph,
    convert_graph_to_gexf,
    convert_graph_to_jsonld,
    create_legends,
)
from src.utils import slugify_filename

try:
    import umap

    umap_installed = True
except ImportError:
    umap_installed = False


_ANNOTATION_COMPONENT_PATH = Path(__file__).parent / "components" / "annotation_editor"
_ANNOTATION_COMPONENT = components.declare_component(
    "annotation_editor",
    path=str(_ANNOTATION_COMPONENT_PATH),
)

_ANNOTATION_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANNOTATION_EVENT_RE = re.compile(r"\son\w+\s*=\s*(['\"]).*?\1", re.IGNORECASE | re.DOTALL)
_ANNOTATION_JS_URL_RE = re.compile(r"href\s*=\s*(['\"])\s*javascript:[^'\"]*\1", re.IGNORECASE)
_ANNOTATION_TAG_RE = re.compile(r"<[^>]+>")
_ANNOTATION_BREAK_RE = re.compile(r"<\s*(br\s*/?|/p\s*|/li\s*)>", re.IGNORECASE)


def _sanitize_annotation_html(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    cleaned = _ANNOTATION_SCRIPT_RE.sub("", cleaned)
    cleaned = _ANNOTATION_EVENT_RE.sub("", cleaned)
    cleaned = _ANNOTATION_JS_URL_RE.sub('href="#"', cleaned)
    return cleaned


def _clean_annotation_text(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _annotation_html_to_text(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _ANNOTATION_BREAK_RE.sub("\n", value)
    cleaned = _ANNOTATION_SCRIPT_RE.sub("", cleaned)
    cleaned = _ANNOTATION_TAG_RE.sub("", cleaned)
    cleaned = html_lib.unescape(cleaned)
    return _clean_annotation_text(cleaned)


def _annotation_text_to_html(value: Optional[str]) -> str:
    cleaned = _clean_annotation_text(value)
    if not cleaned:
        return ""
    return html_lib.escape(cleaned).replace("\n", "<br>")


def _resolve_annotation_state(
    annotation_html: Optional[str],
    annotation_plain: Optional[str],
) -> Tuple[str, str]:
    cleaned_html = _sanitize_annotation_html(annotation_html or "").strip()
    cleaned_plain = _clean_annotation_text(annotation_plain) if isinstance(annotation_plain, str) else ""
    if cleaned_html:
        derived_plain = _annotation_html_to_text(cleaned_html)
        if derived_plain:
            cleaned_plain = derived_plain
    elif cleaned_plain:
        cleaned_html = _annotation_text_to_html(cleaned_plain)
    return cleaned_plain, cleaned_html


def _resolve_annotation_payload(
    payload_html: Optional[str],
    payload_text: Optional[str],
) -> Tuple[str, str]:
    cleaned_html = _sanitize_annotation_html(payload_html or "").strip()
    cleaned_plain = _clean_annotation_text(payload_text) if isinstance(payload_text, str) else ""
    if not cleaned_plain:
        cleaned_plain = _annotation_html_to_text(cleaned_html)
    if cleaned_plain and not cleaned_html:
        cleaned_html = _annotation_text_to_html(cleaned_plain)
    return cleaned_plain, cleaned_html


def _render_annotation_editor(
    value: str,
    placeholder: str,
    key: str,
) -> object:
    return _ANNOTATION_COMPONENT(value=value, placeholder=placeholder, key=key, default={})


def _uri_variants(uri: str) -> List[str]:
    variants: List[str] = []
    if not uri:
        return variants

    def _push(value: Optional[str]) -> None:
        if not value:
            return
        if value not in variants:
            variants.append(value)

    _push(uri)
    cleaned = remove_fragment(uri)
    _push(cleaned)
    if cleaned.endswith("/"):
        _push(cleaned.rstrip("/"))
    else:
        _push(cleaned + "/")
    if cleaned.startswith("https://"):
        _push("http://" + cleaned[len("https://") :])
    elif cleaned.startswith("http://"):
        _push("https://" + cleaned[len("http://") :])
    normalized = _normalize_getty_ulan(cleaned)
    if normalized != cleaned:
        _push(normalized)
    for vocab in ("ulan", "aat", "tgn", "language"):
        base = f"vocab.getty.edu/{vocab}/"
        page = f"vocab.getty.edu/page/{vocab}/"
        if base in cleaned and page not in cleaned:
            _push(cleaned.replace(base, page))
        if page in cleaned and base not in cleaned:
            _push(cleaned.replace(page, base))
    for item in list(variants):
        for ns, prefix in NAMESPACE_PREFIXES.items():
            if item.startswith(ns):
                _push(f"{prefix}:{item[len(ns):]}")
    return variants


def _get_label_lookup() -> Dict[str, str]:
    label_lookup = dict(st.session_state.id_to_label or {})
    base_labels = st.session_state.get("oclc_base_labels")
    if isinstance(base_labels, dict):
        for node_id, label in base_labels.items():
            if not node_id or _label_is_unnamed(label, node_id):
                continue
            if node_id not in label_lookup or _label_is_unnamed(
                label_lookup.get(node_id), node_id
            ):
                label_lookup[node_id] = label

    def _add_label_variant(node_id: str, label: str) -> None:
        if not node_id or _label_is_unnamed(label, node_id):
            return
        existing = label_lookup.get(node_id)
        if not existing or _label_is_unnamed(existing, node_id):
            label_lookup[node_id] = label

    # Augment lookup with current graph data
    if st.session_state.graph_data and st.session_state.graph_data.nodes:
        for n in st.session_state.graph_data.nodes:
            if not n.id:
                continue
            candidates: List[str] = []
            if n.label:
                candidates.append(str(n.label))
            if isinstance(n.metadata, dict):
                pref = n.metadata.get("prefLabel")
                if isinstance(pref, dict):
                    pref_en = pref.get("en")
                    if isinstance(pref_en, str) and pref_en.strip():
                        candidates.append(pref_en.strip())
                    else:
                        for val in pref.values():
                            if isinstance(val, str) and val.strip():
                                candidates.append(val.strip())
                                break
                elif isinstance(pref, str) and pref.strip():
                    candidates.append(pref.strip())
                meta_label = n.metadata.get("_label") or n.metadata.get("label")
                if isinstance(meta_label, str) and meta_label.strip():
                    candidates.append(meta_label.strip())
                name_val = n.metadata.get("name")
                if isinstance(name_val, str) and name_val.strip():
                    candidates.append(name_val.strip())
            for candidate in candidates:
                for variant in _uri_variants(n.id):
                    _add_label_variant(variant, candidate)
    
    return label_lookup


def render_node_details_panel(node_id: str, is_modal: bool = False) -> None:
    if not node_id:
        return
        
    node_obj = next(
        (n for n in st.session_state.graph_data.nodes if n.id == node_id),
        None,
    )
    if not node_obj:
        st.warning(f"Node {node_id} not found in current graph.")
        return

    display_label = st.session_state.id_to_label.get(node_obj.id, node_obj.id)
    if not is_modal:
        st.markdown(f"### {html_lib.escape(display_label)}")
    st.markdown(
        f"<div style='color:#6B7280; font-size:0.85rem;'>ID: <code>{html_lib.escape(node_obj.id)}</code></div>",
        unsafe_allow_html=True,
    )

    canon_types = [canonical_type(t) for t in (node_obj.types or [])]
    st.markdown(
        _render_type_badges(canon_types, st.session_state.node_type_colors),
        unsafe_allow_html=True,
    )

    reserved_keys = {"id", "prefLabel", "type", "inferredTypes", "annotation", "annotation_html"}
    properties = {
        k: v for k, v in (node_obj.metadata or {}).items() if k not in reserved_keys
    }

    m1, m2, m3 = st.columns(3)
    m1.metric("Edges", len(node_obj.edges))
    m2.metric("Properties", len(properties))
    inferred_types = _extract_inferred_types(node_obj.metadata)
    m3.metric("Inferred Types", len(inferred_types))

    annotation_plain = st.session_state.annotations.get(node_obj.id, "")
    annotation_html = st.session_state.annotation_html.get(node_obj.id, "")
    node_metadata = node_obj.metadata if isinstance(node_obj.metadata, dict) else {}
    if not annotation_plain and node_metadata.get("annotation"):
        annotation_plain = str(node_metadata.get("annotation") or "")
    if not annotation_html and node_metadata.get("annotation_html"):
        annotation_html = str(node_metadata.get("annotation_html") or "")
    annotation_plain, annotation_html = _resolve_annotation_state(
        annotation_html, annotation_plain
    )
    if annotation_plain:
        st.session_state.annotations[node_obj.id] = annotation_plain
    if annotation_html:
        st.session_state.annotation_html[node_obj.id] = annotation_html
    if annotation_plain:
        node_metadata["annotation"] = annotation_plain
    if annotation_html:
        node_metadata["annotation_html"] = annotation_html
    if not isinstance(node_obj.metadata, dict):
        node_obj.metadata = node_metadata

    unique_key_suffix = "modal" if is_modal else "panel"
    metadata_tabs = st.tabs(
        ["Overview", "Annotation", "Centrality", "Inferred Types", "Properties", "Raw"]
    )

    with metadata_tabs[0]:
        if annotation_html:
            safe_annotation = _sanitize_annotation_html(annotation_html)
            with st.container(border=True):
                st.markdown(safe_annotation, unsafe_allow_html=True)
        if annotation_plain:
            st.info(annotation_plain)
        else:
            st.caption("No annotation for this node yet.")
        
        # Use shared helper for label lookup
        label_lookup = _get_label_lookup()

        # Custom rendering for specific complex keys to avoid CSS modal issues
        # 1. Conservation Actions (583)
        cons_keys = ["583", "conservationActions", "conservation_actions"]
        cons_actions = []
        for k in cons_keys:
            if k in (node_obj.metadata or {}):
                val = node_obj.metadata[k]
                cons_actions.extend(_coerce_list(val))
        
        if cons_actions:
            st.markdown("#### Conservation Actions (MARC 583)")
            # Use expander for the list if it's long, or just render if short? 
            # Actually, user wants to see them. Let's use a container style.
            # But specific complaint was about the modal.
            
            # Use native Streamlit expanders for each action if valid
            for idx, action in enumerate(cons_actions):
                # Reuse the util logic for title/summary but render natively
                if isinstance(action, dict):
                    title_raw = action.get("action") or action.get("status") or "Action"
                    title = _truncate_text(str(title_raw), 80)
                    # Get the body HTML from the util, but we can't strip the outer div easily maybe.
                    # Actually, _format_conservation_action_block returns an HTML block <div class='meta-note'>...</div>
                    # We can just render that inside the loop.
                    action_html = _format_conservation_action_block(action)
                    st.markdown(action_html, unsafe_allow_html=True)
                else:
                    st.markdown(f"- {action}")
            
            st.markdown("---")


        st.markdown("#### Metadata Overview")
        with st.container(border=True):
            meta_overview_html = _render_metadata_overview(
                node_obj.metadata or {},
                skip_keys=reserved_keys | set(cons_keys),
                max_cards=12,
                label_lookup=label_lookup,
            )
            if meta_overview_html:
                st.markdown(meta_overview_html, unsafe_allow_html=True)
            else:
                st.caption("No additional metadata available.")

    with metadata_tabs[1]:
        if not st.session_state.graph_data.nodes:
            st.caption("No nodes available for annotation.")
        else:
            status_key = f"annotation_status_{node_obj.id}"
            status = st.session_state.pop(status_key, None)
            if status == "saved":
                st.success("Annotation saved.")
            elif status == "cleared":
                st.info("Annotation cleared.")

            editor_event = _render_annotation_editor(
                annotation_html,
                "Add a rich-text annotation for this node.",
                key=f"annotation_editor_{node_obj.id}_{unique_key_suffix}",
            )
            action = None
            payload_html = None
            payload_text = None
            nonce = None
            if isinstance(editor_event, dict):
                action = editor_event.get("action")
                payload_html = editor_event.get("html")
                payload_text = editor_event.get("text")
                nonce = editor_event.get("nonce")
            elif isinstance(editor_event, str):
                action = "save"
                payload_html = editor_event
            if action in {"save", "clear"}:
                nonce_key = f"annotation_event_nonce_{node_obj.id}_{unique_key_suffix}"
                last_nonce = st.session_state.get(nonce_key)
                if nonce and nonce == last_nonce:
                    pass
                else:
                    if nonce:
                        st.session_state[nonce_key] = nonce
                    cleaned_plain, cleaned_html = _resolve_annotation_payload(
                        payload_html, payload_text
                    )
                    if action == "clear" or not cleaned_plain:
                        if node_obj.id in st.session_state.annotation_html:
                            st.session_state.annotation_html.pop(node_obj.id, None)
                        if node_obj.id in st.session_state.annotations:
                            st.session_state.annotations.pop(node_obj.id, None)
                        if isinstance(node_obj.metadata, dict):
                            node_obj.metadata.pop("annotation_html", None)
                            node_obj.metadata.pop("annotation", None)
                        st.session_state[status_key] = "cleared"
                        st.rerun()
                    else:
                        current_html = st.session_state.annotation_html.get(node_obj.id, "")
                        current_plain = st.session_state.annotations.get(node_obj.id, "")
                        if cleaned_html != current_html or cleaned_plain != current_plain:
                            st.session_state.annotation_html[node_obj.id] = cleaned_html
                            st.session_state.annotations[node_obj.id] = cleaned_plain
                            if isinstance(node_obj.metadata, dict):
                                node_obj.metadata["annotation_html"] = cleaned_html
                                node_obj.metadata["annotation"] = cleaned_plain
                            st.session_state[status_key] = "saved"
                            st.rerun()

    with metadata_tabs[2]:
        if (
            st.session_state.centrality_measures
            and node_obj.id in st.session_state.centrality_measures
        ):
            c_meas = st.session_state.centrality_measures[node_obj.id]
            st.markdown(
                f"- **Degree:** {c_meas['degree']:.3f}\n"
                f"- **Betweenness:** {c_meas['betweenness']:.3f}\n"
                f"- **Closeness:** {c_meas['closeness']:.3f}\n"
                f"- **Eigenvector:** {c_meas['eigenvector']:.3f}\n"
                f"- **PageRank:** {c_meas['pagerank']:.3f}"
            )
        else:
            st.caption("No centrality metrics available yet.")

    with metadata_tabs[3]:
        if inferred_types:
            mapped_types = sorted({canonical_type(t) for t in inferred_types if canonical_type(t) != "Unknown"})
            if mapped_types:
                st.markdown(f"**Mapped:** {', '.join(mapped_types)}")
            st.markdown("\n".join([f"- {_shorten_iri(t)}" for t in inferred_types]))
        else:
            st.caption("No inferred types available for this node.")

    with metadata_tabs[4]:
        if properties:
            prop_rows = _metadata_rows(node_obj.metadata, skip_keys=reserved_keys)
            prop_df = pd.DataFrame(prop_rows)
            st.dataframe(prop_df, use_container_width=True, height=260)
        else:
            st.caption("No additional properties available for this node.")

    with metadata_tabs[5]:
        st.json(node_obj.metadata)


@st.dialog("Node Details", width="large")
def node_details_modal(node_id: str):
    render_node_details_panel(node_id, is_modal=True)


def render_tabs(
    filtered_nodes: Optional[Set[str]],
    community_detection: bool,
    effective_node_animations: List[str],
) -> None:
    tabs = st.tabs(
        [
            "Graph View",
            "Conservation",
            "Data View",
            "Centrality Measures",
            "SPARQL Query",
            "Timeline",
            "Graph Embeddings",
            "Node Similarity Search",
            "About",
        ]
    )

    # Check for modal trigger
    if st.session_state.modal_node:
        node_details_modal(st.session_state.modal_node)
        # We don't clear it immediately to avoid closing it on header re-renders, 
        # but logically st.dialog handles the close action. 
        # However, if we want it to close cleanly, we might need to handle the close event, 
        # but st.dialog handles the overlay state. 
        # Actually, st.dialog is a declaration. Calling it opens it. 
        # It stays open until user closes.
        # We might want to clear the state if we can detect close, but for now this suffices.


    with tabs[0]:
        st.header("Network Graph")
        if st.session_state.graph_data.nodes:
            with st.spinner("Generating Network Graph..."):
                search_nodes = st.session_state.search_nodes
                if not st.session_state.search_term.strip():
                    search_nodes = None

                if st.session_state.property_filter["property"] and st.session_state.property_filter["value"]:
                    prop = st.session_state.property_filter["property"]
                    val = st.session_state.property_filter["value"].lower()
                    prop_nodes = {
                        n.id
                        for n in st.session_state.graph_data.nodes
                        if prop in n.metadata and val in str(n.metadata[prop]).lower()
                    }
                    filtered_nodes = (
                        filtered_nodes.intersection(prop_nodes) if filtered_nodes is not None else prop_nodes
                    )

                candidate_nodes = {n.id for n in st.session_state.graph_data.nodes}
                if filtered_nodes is not None:
                    candidate_nodes = candidate_nodes.intersection(filtered_nodes)
                protected_nodes = []
                if search_nodes:
                    protected_nodes.extend(search_nodes)
                if st.session_state.shortest_path:
                    protected_nodes.extend(st.session_state.shortest_path)
                if st.session_state.selected_node:
                    protected_nodes.append(st.session_state.selected_node)
                capped_nodes, cap_stats = apply_render_cap(
                    st.session_state.graph_data,
                    candidate_nodes,
                    MAX_RENDER_NODES,
                    st.session_state.selected_relationships,
                    protected_nodes,
                )
                if cap_stats["trimmed"] > 0:
                    filtered_nodes = capped_nodes
                    st.info(f"Render cap applied: showing {len(capped_nodes)} of {cap_stats['total']} nodes.")
                    if cap_stats["protected_trimmed"] > 0:
                        st.warning("Some prioritized nodes were omitted due to the render cap.")

                physics_params = st.session_state.get("physics_params", {})
                physics_signature = tuple(
                    (key, round(float(value), 4)) for key, value in sorted(physics_params.items())
                )
                search_signature = st.session_state.search_term.strip().lower()
                path_signature = tuple(st.session_state.shortest_path or [])
                animation_signature = tuple(sorted(effective_node_animations))
                render_node_count = (
                    len(filtered_nodes) if filtered_nodes is not None else len(st.session_state.graph_data.nodes)
                )
                graph_version = hash(
                    (
                        len(st.session_state.graph_data.nodes),
                        sum(len(n.edges) for n in st.session_state.graph_data.nodes),
                        render_node_count,
                        MAX_RENDER_NODES,
                        int(st.session_state.performance_mode),
                        st.session_state.get("physics_preset", "Default (Balanced)"),
                        physics_signature,
                        int(st.session_state.show_labels),
                        int(st.session_state.smart_labels),
                        float(st.session_state.label_zoom_threshold),
                        int(st.session_state.reduce_motion),
                        int(st.session_state.motion_intensity),
                        animation_signature,
                        int(st.session_state.node_animation_strength),
                        int(st.session_state.focus_context),
                        int(st.session_state.edge_semantics),
                        int(st.session_state.type_icons),
                        search_signature,
                        path_signature,
                    )
                )

                node_type_colors = st.session_state.get("node_type_colors") or CONFIG["NODE_TYPE_COLORS"]
                relationship_colors = st.session_state.get("relationship_colors") or CONFIG["RELATIONSHIP_CONFIG"]
                graph_background = st.session_state.get("graph_background") or CONFIG.get(
                    "GRAPH_BACKGROUND", {}
                )
                community_colors = st.session_state.get("community_colors") or CONFIG.get(
                    "COMMUNITY_COLORS", []
                )
                community_labels = st.session_state.get("community_labels") or [
                    f"Community {idx + 1}" for idx in range(len(community_colors))
                ]
                community_locks = st.session_state.get("community_locks") or {}

                net = build_graph(
                    graph_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_nodes=search_nodes,
                    node_positions=st.session_state.node_positions,
                    show_labels=st.session_state.show_labels,
                    smart_labels=st.session_state.smart_labels,
                    label_zoom_threshold=st.session_state.label_zoom_threshold,
                    filtered_nodes=filtered_nodes,
                    community_detection=community_detection,
                    centrality=st.session_state.centrality_measures,
                    path_nodes=st.session_state.shortest_path,
                    graph_version=graph_version,
                    reduce_motion=st.session_state.reduce_motion,
                    motion_intensity=st.session_state.motion_intensity,
                    node_animations=effective_node_animations,
                    node_animation_strength=st.session_state.node_animation_strength,
                    node_type_colors=node_type_colors,
                    relationship_colors=relationship_colors,
                    graph_background=graph_background,
                    community_colors=community_colors,
                    community_locks=community_locks,
                    focus_context=st.session_state.focus_context,
                    edge_semantics=st.session_state.edge_semantics,
                    type_icons=st.session_state.type_icons,
                )

            if community_detection and getattr(net, "_community_applied", False):
                st.info("Community detection applied to node colors.")

            if len(net.nodes) > 50 and st.session_state.show_labels:
                st.info("Graph has many nodes. Consider toggling 'Show Node Labels' off for better readability.")

            node_total = len(net.nodes)
            edge_total = len(net.edges)
            rel_total = len({edge.get("rel_key") for edge in net.edges if edge.get("rel_key")})
            type_total = len({canonical_type(t) for n in st.session_state.graph_data.nodes for t in n.types})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nodes", node_total)
            m2.metric("Edges", edge_total)
            m3.metric("Relationships", rel_total)
            m4.metric("Node Types", type_total)

            try:
                graph_signature = (
                    f"graph-{graph_version}-"
                    f"{int(st.session_state.reduce_motion)}-"
                    f"{st.session_state.motion_intensity}-"
                    f"{st.session_state.get('physics_preset', 'Default (Balanced)')}-"
                    f"{'-'.join(animation_signature) if animation_signature else 'none'}-"
                    f"{st.session_state.node_animation_strength}"
                )
                html_code = net.html
                if "</body>" in html_code:
                    html_code = html_code.replace(
                        "</body>",
                        (
                            "<div style='display:none' "
                            f"data-graph-signature='{graph_signature}'></div></body>"
                        ),
                        1,
                    )
                else:
                    html_code += f"<!-- {graph_signature} -->"
                components.html(html_code, height=GRAPH_CARD_HEIGHT, scrolling=False)
            except Exception as exc:
                st.error(f"Graph generation failed: {exc}")

            with st.expander("Legends", expanded=False):
                legend_community_colors = None
                legend_community_labels = None
                if community_detection and getattr(net, "_community_applied", False):
                    legend_community_colors = getattr(net, "_community_palette", community_colors)
                    community_count = getattr(net, "_community_count", len(legend_community_colors))
                    legend_community_colors = legend_community_colors[:community_count]
                    legend_community_labels = list(st.session_state.get("community_labels") or [])
                    if len(legend_community_labels) < community_count:
                        legend_community_labels = legend_community_labels + [
                            f"Community {idx + 1}"
                            for idx in range(len(legend_community_labels), community_count)
                        ]
                    legend_community_labels = legend_community_labels[:community_count]
                    lock_labels = st.session_state.get("community_lock_labels") or {}
                    community_locks = st.session_state.get("community_locks") or {}
                    for slot_idx, node_id in community_locks.items():
                        if slot_idx < len(legend_community_labels) and node_id in lock_labels:
                            legend_community_labels[slot_idx] = lock_labels[node_id]
                st.markdown(
                    create_legends(
                        relationship_colors,
                        node_type_colors,
                        legend_community_colors,
                        legend_community_labels,
                    ),
                    unsafe_allow_html=True,
                )
                if st.session_state.edge_semantics:
                    has_inferred = any(
                        getattr(e, "inferred", False)
                        for n in st.session_state.graph_data.nodes
                        for e in n.edges
                    )
                    has_weighted = any(
                        getattr(e, "weight", None) is not None
                        for n in st.session_state.graph_data.nodes
                        for e in n.edges
                    )
                    if has_inferred:
                        st.caption("Dashed edges indicate inferred relationships.")
                    if has_weighted:
                        st.caption("Thicker edges indicate higher confidence or weight when available.")
            with st.expander("Node Metadata", expanded=False):
                if st.session_state.graph_data.nodes:
                    node_options = {
                        n.id: st.session_state.id_to_label.get(n.id, n.id)
                        for n in st.session_state.graph_data.nodes
                    }
                    node_ids = list(node_options.keys())
                    default_index = 0
                    if st.session_state.selected_node in node_options:
                        default_index = node_ids.index(st.session_state.selected_node)
                    selected_node_ui = st.selectbox(
                        "Select Node",
                        options=node_ids,
                        index=default_index,
                        format_func=lambda x: node_options.get(x, x),
                        key="metadata_node_select",
                    )
                    if selected_node_ui:
                        st.session_state.selected_node = selected_node_ui
                        if st.session_state.selected_node:
                            render_node_details_panel(st.session_state.selected_node, is_modal=False)
                        else:
                            st.info("Select a node to view its metadata.")

            with st.expander("IIIF Viewer", expanded=False):
                def _normalize_manifest_url(value: Any) -> str:
                    if not value:
                        return ""
                    if isinstance(value, list):
                        value = value[0] if value else ""
                    if not isinstance(value, str):
                        return ""
                    cleaned = value.strip()
                    if not cleaned:
                        return ""
                    parsed = urlparse(cleaned)
                    query_params = parse_qs(parsed.query)
                    if "manifest" in query_params:
                        cleaned = query_params["manifest"][0]
                    return cleaned.strip()

                manual_manifest = st.text_input(
                    "Manual IIIF manifest URL",
                    help="Paste a manifest URL to preview without adding it to a node.",
                    placeholder="https://example.org/iiif/manifest",
                    key="iiif_manifest_manual",
                )
                manual_manifest = _normalize_manifest_url(manual_manifest)

                all_nodes = st.session_state.graph_data.nodes if st.session_state.graph_data else []
                if all_nodes:
                    node_options = {
                        n.id: st.session_state.id_to_label.get(n.id, n.label or n.id)
                        for n in all_nodes
                        if n.id
                    }
                    if not node_options:
                        st.caption("No nodes available to save a manifest.")
                    else:
                        default_index = 0
                        if st.session_state.get("selected_node") in node_options:
                            default_index = list(node_options.keys()).index(st.session_state.selected_node)
                        else:
                            # If selected node not in graph (rare), default to 0
                            default_index = 0

                        save_target = st.selectbox(
                            "Select node to attach manifest",
                            options=list(node_options.keys()),
                            index=default_index,
                            format_func=lambda x: node_options.get(x, x),
                            key="iiif_manifest_save_node",
                        )

                        # Show existing manifest if any
                        target_node_obj = next((n for n in all_nodes if n.id == save_target), None)
                        if target_node_obj and isinstance(target_node_obj.metadata, dict):
                            existing_manifest = target_node_obj.metadata.get("manifest")
                            if existing_manifest:
                                st.info(f"Node has manifest: `{existing_manifest}`")

                        c1, c2 = st.columns([1, 1])
                        with c1:
                            save_manifest = st.button(
                                "Save manifest to node",
                                disabled=not manual_manifest,
                                help="Associates the URL above with the selected node.",
                                use_container_width=True,
                            )
                        with c2:
                            if target_node_obj and isinstance(target_node_obj.metadata, dict) and target_node_obj.metadata.get("manifest"):
                                if st.button("Clear manifest", type="primary", use_container_width=True):
                                    target_node_obj.metadata.pop("manifest", None)
                                    st.success("Manifest removed from node.")
                                    st.rerun()

                        if save_manifest:
                            if not manual_manifest:
                                st.warning("Enter a manifest URL before saving.")
                            else:
                                if target_node_obj:
                                    # Find the node in the actual session state list to ensure we update the source of truth
                                    found_index = next(
                                        (i for i, n in enumerate(st.session_state.graph_data.nodes) if n.id == target_node_obj.id),
                                        None
                                    )
                                    if found_index is not None:
                                        node_ref = st.session_state.graph_data.nodes[found_index]
                                        if not isinstance(node_ref.metadata, dict):
                                            node_ref.metadata = {}
                                        node_ref.metadata["manifest"] = manual_manifest
                                        
                                        # Also update the local reference just in case
                                        target_node_obj.metadata = node_ref.metadata
                                        
                                        st.success(f"Manifest URL saved to node '{st.session_state.id_to_label.get(node_ref.id, node_ref.id)}'.")
                                        if not is_valid_iiif_manifest(manual_manifest):
                                            st.warning(
                                                "This URL doesn't look like a IIIF manifest; the viewer may not load it."
                                            )
                                        st.caption("Changes are in memory. Remember to 'Save Workspace' in the sidebar to keep them.")
                                        time.sleep(1.0) # Give user time to see message
                                        st.rerun()
                                    else:
                                        st.error("Node not found in graph data.")
                                else:
                                    st.warning("Select a node before saving.")
                iiif_nodes = [
                    n
                    for n in st.session_state.graph_data.nodes
                    if isinstance(n.metadata, dict) and ("image" in n.metadata or "manifest" in n.metadata)
                ]
                manifest_url = ""
                if manual_manifest:
                    manifest_url = manual_manifest
                elif iiif_nodes:
                    sel_iiif = st.selectbox(
                        "Select an entity with a manifest for IIIF Viewer",
                        [n.id for n in iiif_nodes],
                        format_func=lambda x: st.session_state.id_to_label.get(x, x),
                    )
                    node_iiif = next((n for n in iiif_nodes if n.id == sel_iiif), None)
                    if node_iiif:
                        manifest_url = _normalize_manifest_url(
                            node_iiif.metadata.get("image") or node_iiif.metadata.get("manifest")
                        )

                if manifest_url:
                    if is_valid_iiif_manifest(manifest_url):
                        st.write(f"Using manifest URL: {manifest_url}")
                        html_code = f'''
<html>
  <head>
    <link rel="stylesheet" href="https://unpkg.com/mirador@3.4.3/dist/mirador.min.css"/>
    <script src="https://unpkg.com/mirador@3.4.3/dist/mirador.min.js"></script>
  </head>
  <body>
    <div id="mirador-viewer" style="height: 600px;"></div>
    <script>
      var manifestUrl = {json.dumps(manifest_url)};
      console.log("Manifest URL:", manifestUrl);
      Mirador.viewer({{
        id: 'mirador-viewer',
        windows: [{{ "loadedManifest": manifestUrl }}]
      }});
    </script>
  </body>
</html>
'''
                        components.html(html_code, height=650)
                    else:
                        st.info("No valid IIIF manifest found for the selected URL.")
                elif not iiif_nodes:
                    st.info("No entity with a manifest found.")

            with st.expander("Map View of Entities with Coordinates", expanded=False):
                def _extract_lat_lon(node: Node) -> Optional[Tuple[float, float]]:
                    if not isinstance(node.metadata, dict):
                        return None
                    coords = node.metadata.get("geographicCoordinates")
                    if coords:
                        if isinstance(coords, list):
                            coords = coords[0]
                        if isinstance(coords, str) and coords.startswith("Point(") and coords.endswith(")"):
                            coords = coords[6:-1].strip()
                            parts = coords.split()
                            if len(parts) == 2:
                                try:
                                    lon, lat = map(float, parts)
                                    return lat, lon
                                except ValueError:
                                    logging.error("Invalid coordinates for node %s: %s", node.id, coords)
                    lat = node.metadata.get("latitude")
                    lon = node.metadata.get("longitude")
                    if lat is not None and lon is not None:
                        try:
                            return float(lat), float(lon)
                        except ValueError:
                            logging.error("Invalid numeric coordinates for node %s: %s, %s", node.id, lat, lon)
                    return None

                place_locations = []
                coord_map: Dict[str, Tuple[float, float]] = {}
                for n in st.session_state.graph_data.nodes:
                    lat_lon = _extract_lat_lon(n)
                    if lat_lon:
                        lat, lon = lat_lon
                        coord_map[n.id] = (lat, lon)
                        place_locations.append({"lat": lat, "lon": lon, "label": n.label, "id": n.id})
                if place_locations:
                    df_places = pd.DataFrame(place_locations)
                    st.map(df_places)

                    st.subheader("PyDeck Arc Map (Geospatial Relationships)")
                    arc_rows = []
                    seen_edges: Set[Tuple[str, str, str]] = set()
                    for n in st.session_state.graph_data.nodes:
                        for e in n.edges:
                            key = (e.source, e.target, e.relationship)
                            if key in seen_edges:
                                continue
                            if e.source in coord_map and e.target in coord_map:
                                src_lat, src_lon = coord_map[e.source]
                                tgt_lat, tgt_lon = coord_map[e.target]
                                arc_rows.append(
                                    {
                                        "source_lat": src_lat,
                                        "source_lon": src_lon,
                                        "target_lat": tgt_lat,
                                        "target_lon": tgt_lon,
                                        "source_label": st.session_state.id_to_label.get(e.source, e.source),
                                        "target_label": st.session_state.id_to_label.get(e.target, e.target),
                                        "relationship": e.relationship,
                                    }
                                )
                                seen_edges.add(key)

                    if arc_rows:
                        df_arcs = pd.DataFrame(arc_rows)
                        st.slider(
                            "Arc map zoom",
                            min_value=0.5,
                            max_value=10.0,
                            step=0.5,
                            key="arc_map_zoom",
                        )
                        st.slider(
                            "Arc map pitch",
                            min_value=0,
                            max_value=60,
                            step=5,
                            key="arc_map_pitch",
                        )
                        view_state = pdk.ViewState(
                            latitude=float(df_places["lat"].mean()),
                            longitude=float(df_places["lon"].mean()),
                            zoom=float(st.session_state.arc_map_zoom),
                            pitch=int(st.session_state.arc_map_pitch),
                        )
                        arc_layer = pdk.Layer(
                            "ArcLayer",
                            data=df_arcs,
                            get_source_position=["source_lon", "source_lat"],
                            get_target_position=["target_lon", "target_lat"],
                            get_source_color=[208, 106, 76],
                            get_target_color=[45, 79, 106],
                            get_width=2,
                            auto_highlight=True,
                            pickable=True,
                        )
                        deck = pdk.Deck(
                            layers=[arc_layer],
                            initial_view_state=view_state,
                            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                            tooltip={"text": "{source_label} -> {target_label}\n{relationship}"},
                        )
                        deck_html = deck.to_html(as_string=True, iframe_height=520, iframe_width="100%")
                        components.html(deck_html, height=520, scrolling=False)
                    else:
                        st.caption("No geospatial relationships found between mapped entities.")
                else:
                    st.info("No entities with valid coordinates found for map view.")

            with st.expander("Shortest Path Details", expanded=False):
                if st.session_state.shortest_path:
                    path_list = st.session_state.shortest_path
                    path_text = " -> ".join([st.session_state.id_to_label.get(x, x) for x in path_list])
                    path_text = re.sub(r"[^\x20-\x7E]+", "", path_text)
                    if len(path_text) > 1000:
                        path_text = path_text[:1000] + "... [truncated]"
                    st.text_area("Shortest Path", value=path_text, height=100)

                    rel_text = ""
                    for i in range(len(path_list) - 1):
                        rels = get_edge_relationship(path_list[i], path_list[i + 1], st.session_state.graph_data)
                        rel_text += (
                            f"{st.session_state.id_to_label.get(path_list[i], path_list[i])} -- {', '.join(rels)} --> "
                        )
                    rel_text += st.session_state.id_to_label.get(path_list[-1], path_list[-1])
                    rel_text = re.sub(r"[^\x20-\x7E]+", "", rel_text)
                    if len(rel_text) > 1000:
                        rel_text = rel_text[:1000] + "... [truncated]"
                    st.text_area("Path Relationships", value=rel_text, height=100)
                else:
                    st.info("No shortest path available.")

            with st.expander("Export Options", expanded=False):
                st.session_state.export_title_draft = st.text_input(
                    "Export Title",
                    value=st.session_state.export_title_draft,
                    help="Used for the title in the downloaded HTML export.",
                )
                title_saved = st.button("Apply Export Title")
                if title_saved:
                    st.session_state.export_title = (
                        st.session_state.export_title_draft.strip() or "Linked Data Explorer"
                    )
                    st.success("Export title updated.")
                export_community_labels = list(community_labels)
                community_count = getattr(net, "_community_count", len(community_colors))
                if len(export_community_labels) < community_count:
                    export_community_labels.extend(
                        f"Community {idx + 1}"
                        for idx in range(len(export_community_labels), community_count)
                    )
                export_community_labels = export_community_labels[:community_count]
                lock_labels = st.session_state.get("community_lock_labels") or {}
                community_locks = st.session_state.get("community_locks") or {}
                for slot_idx, node_id in community_locks.items():
                    if slot_idx < len(export_community_labels) and node_id in lock_labels:
                        export_community_labels[slot_idx] = lock_labels[node_id]
                export_label_lookup = dict(st.session_state.id_to_label or {})
                base_labels = st.session_state.get("oclc_base_labels")
                if isinstance(base_labels, dict):
                    for node_id, label in base_labels.items():
                        if not node_id or _label_is_unnamed(label, node_id):
                            continue
                        existing = export_label_lookup.get(node_id)
                        if not existing or _label_is_unnamed(existing, node_id):
                            export_label_lookup[node_id] = label
                export_html = build_export_html(
                    net,
                    st.session_state.graph_data,
                    relationship_colors,
                    node_type_colors,
                    st.session_state.export_title,
                    id_to_label=export_label_lookup,
                    reduce_motion=st.session_state.reduce_motion,
                    node_animations=effective_node_animations,
                    node_animation_strength=st.session_state.node_animation_strength,
                    search_nodes=search_nodes,
                    path_nodes=st.session_state.shortest_path,
                    centrality_measures=st.session_state.centrality_measures,
                    graph_background=graph_background,
                    community_colors=community_colors,
                    community_labels=export_community_labels,
                    show_labels=st.session_state.show_labels,
                    smart_labels=st.session_state.smart_labels,
                    label_zoom_threshold=st.session_state.label_zoom_threshold,
                    focus_context=st.session_state.focus_context,
                )
                export_slug = slugify_filename(st.session_state.export_title)
                export_filename = f"{export_slug}.html"
                st.session_state.graph_html = export_html
                if "graph_html" in st.session_state:
                    st.download_button(
                        "Download Graph as HTML",
                        data=st.session_state.graph_html,
                        file_name=export_filename,
                        mime="text/html",
                    )
                gexf_data = convert_graph_to_gexf(st.session_state.graph_data, st.session_state.id_to_label)
                st.download_button(
                    "Download Graph as GEXF",
                    data=gexf_data,
                    file_name=f"{export_slug}.gexf",
                    mime="application/gexf+xml",
                )
                jsonld_data = convert_graph_to_jsonld(net)
                jsonld_str = json.dumps(jsonld_data, indent=2)
                st.download_button(
                    "Download Graph Data as JSON-LD",
                    data=jsonld_str,
                    file_name="graph_data.jsonld",
                    mime="application/ld+json",
                )
        else:
            st.info("No valid data found. Please check your JSON/RDF files.")

    with tabs[1]:
        st.header("Conservation Mode")
        if not st.session_state.graph_data.nodes:
            st.info("Load a MARC21 .dat file or other data to view conservation actions.")
        else:
            def _is_work_node(node: Node) -> bool:
                canon_types = [canonical_type(t) for t in (node.types or [])]
                return any(t in {"Work", "StillImage"} for t in canon_types)

            work_nodes = [n for n in st.session_state.graph_data.nodes if _is_work_node(n)]
            if not work_nodes:
                st.info("No Work nodes available yet.")
            else:
                work_options = {
                    n.id: st.session_state.id_to_label.get(n.id, n.label or n.id)
                    for n in work_nodes
                }
                selected_work = st.selectbox(
                    "Select Work",
                    options=list(work_options.keys()),
                    format_func=lambda x: work_options.get(x, x),
                    key="conservation_work_select",
                )

                work_node = next(
                    (n for n in st.session_state.graph_data.nodes if n.id == selected_work), None
                )
                def _normalize_label(value: str) -> str:
                    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", value.lower())
                    return re.sub(r"\s+", " ", cleaned).strip()

                def _split_agents(raw: str) -> List[str]:
                    if not raw:
                        return []
                    normalized = " ".join(raw.split())
                    if ";" not in normalized:
                        return [normalized] if normalized else []
                    parts = [part.strip() for part in normalized.split(";") if part.strip()]
                    return parts

                if work_node and isinstance(work_node.metadata, dict):
                    actions = work_node.metadata.get("583")
                    st.markdown("#### Existing Actions (MARC 583)")
                    if actions:
                        st.markdown(
                            _render_metadata_overview({"583": actions}, max_cards=3),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No conservation actions for this work yet.")

                st.markdown("#### Work Annotation")
                if work_node:
                    status_key = f"work_annotation_status_{work_node.id}"
                    status = st.session_state.pop(status_key, None)
                    if status == "saved":
                        st.success("Work annotation saved.")
                    elif status == "cleared":
                        st.info("Work annotation cleared.")

                    work_annotation_html = st.session_state.annotation_html.get(work_node.id, "")
                    work_annotation_plain = st.session_state.annotations.get(work_node.id, "")
                    if not work_annotation_plain and work_node.metadata.get("annotation"):
                        work_annotation_plain = str(work_node.metadata.get("annotation") or "")
                    if not work_annotation_html and work_node.metadata.get("annotation_html"):
                        work_annotation_html = str(work_node.metadata.get("annotation_html") or "")
                    work_annotation_plain, work_annotation_html = _resolve_annotation_state(
                        work_annotation_html, work_annotation_plain
                    )
                    if work_annotation_plain:
                        st.session_state.annotations[work_node.id] = work_annotation_plain
                        work_node.metadata["annotation"] = work_annotation_plain
                    if work_annotation_html:
                        st.session_state.annotation_html[work_node.id] = work_annotation_html
                        work_node.metadata["annotation_html"] = work_annotation_html
                    work_note_event = _render_annotation_editor(
                        work_annotation_html,
                        "Add an annotation for this work.",
                        key=f"work_annotation_editor_{work_node.id}",
                    )
                    if isinstance(work_note_event, dict) and work_note_event:
                        nonce_key = f"work_annotation_nonce_{work_node.id}"
                        if work_note_event.get("nonce") != st.session_state.get(nonce_key):
                            st.session_state[nonce_key] = work_note_event.get("nonce")
                            if work_note_event.get("action") == "clear":
                                st.session_state.annotation_html.pop(work_node.id, None)
                                st.session_state.annotations.pop(work_node.id, None)
                                work_node.metadata.pop("annotation_html", None)
                                work_node.metadata.pop("annotation", None)
                                st.session_state[status_key] = "cleared"
                                st.rerun()
                            elif work_note_event.get("action") == "save":
                                cleaned_plain, cleaned_html = _resolve_annotation_payload(
                                    work_note_event.get("html"),
                                    work_note_event.get("text"),
                                )
                                if not cleaned_plain:
                                    st.session_state.annotation_html.pop(work_node.id, None)
                                    st.session_state.annotations.pop(work_node.id, None)
                                    work_node.metadata.pop("annotation_html", None)
                                    work_node.metadata.pop("annotation", None)
                                    st.session_state[status_key] = "cleared"
                                    st.rerun()
                                st.session_state.annotation_html[work_node.id] = cleaned_html
                                st.session_state.annotations[work_node.id] = cleaned_plain
                                work_node.metadata["annotation_html"] = cleaned_html
                                work_node.metadata["annotation"] = cleaned_plain
                                st.session_state[status_key] = "saved"
                                st.rerun()
                    if work_annotation_html:
                        safe_annotation = _sanitize_annotation_html(work_annotation_html)
                        st.markdown(
                            "<div class='annotation-preview'>"
                            f"{safe_annotation}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif work_annotation_plain:
                        st.info(work_annotation_plain)
                    else:
                        st.caption("No annotation saved yet.")

                st.markdown("---")
                st.subheader("Edit Existing Action")
                action_nodes: List[Node] = []
                if work_node:
                    action_targets = {
                        edge.target
                        for edge in work_node.edges
                        if edge.relationship == "conservationAction"
                    }
                    action_nodes = [
                        node
                        for node in st.session_state.graph_data.nodes
                        if node.id in action_targets
                    ]
                if not action_nodes:
                    st.caption("No conservation actions available to edit yet.")
                else:
                    action_options = {
                        n.id: st.session_state.id_to_label.get(n.id, n.label or n.id)
                        for n in action_nodes
                    }
                    edit_action_id = st.selectbox(
                        "Select Action to Edit",
                        options=list(action_options.keys()),
                        format_func=lambda x: action_options.get(x, x),
                        key="conservation_edit_action_select",
                    )
                    edit_node = next((n for n in action_nodes if n.id == edit_action_id), None)

                    def _value_to_text(value: Any) -> str:
                        if value is None:
                            return ""
                        if isinstance(value, list):
                            return "; ".join(str(v) for v in value if v)
                        return str(value)

                    if edit_node and isinstance(edit_node.metadata, dict):
                        edit_meta = edit_node.metadata
                        if st.session_state.get("conservation_edit_action_last") != edit_node.id:
                            st.session_state.conservation_edit_action_last = edit_node.id
                            st.session_state.conservation_edit_action = _value_to_text(
                                edit_meta.get("action")
                            )
                            st.session_state.conservation_edit_date = _value_to_text(
                                edit_meta.get("date")
                            )
                            st.session_state.conservation_edit_status = _value_to_text(
                                edit_meta.get("status")
                            )
                            st.session_state.conservation_edit_agents = _value_to_text(
                                edit_meta.get("agent")
                            )
                            st.session_state.conservation_edit_method = _value_to_text(
                                edit_meta.get("method")
                            )
                            st.session_state.conservation_edit_materials = _value_to_text(
                                edit_meta.get("materials")
                            )
                            st.session_state.conservation_edit_extent = _value_to_text(
                                edit_meta.get("extent")
                            )
                            st.session_state.conservation_edit_institution = _value_to_text(
                                edit_meta.get("institution")
                            )
                            edit_note_plain, edit_note_html = _resolve_annotation_state(
                                _value_to_text(edit_meta.get("annotation_html")),
                                _value_to_text(edit_meta.get("annotation")),
                            )
                            st.session_state.conservation_edit_note_html = edit_note_html
                            st.session_state.conservation_edit_note_plain = edit_note_plain

                        status_key = f"conservation_edit_note_status_{edit_node.id}"
                        status = st.session_state.pop(status_key, None)
                        if status == "saved":
                            st.success("Action annotation saved.")
                        elif status == "cleared":
                            st.info("Action annotation cleared.")

                        edit_note_html = st.session_state.get("conservation_edit_note_html", "")
                        edit_note_event = _render_annotation_editor(
                            edit_note_html,
                            "Edit action annotation.",
                            key=f"conservation_edit_note_{edit_node.id}",
                        )
                        if isinstance(edit_note_event, dict) and edit_note_event:
                            nonce_key = f"conservation_edit_note_nonce_{edit_node.id}"
                            if edit_note_event.get("nonce") != st.session_state.get(nonce_key):
                                st.session_state[nonce_key] = edit_note_event.get("nonce")
                                if edit_note_event.get("action") == "clear":
                                    st.session_state.conservation_edit_note_html = ""
                                    st.session_state.conservation_edit_note_plain = ""
                                    st.session_state[status_key] = "cleared"
                                    st.rerun()
                                elif edit_note_event.get("action") == "save":
                                    cleaned_plain, cleaned_html = _resolve_annotation_payload(
                                        edit_note_event.get("html"),
                                        edit_note_event.get("text"),
                                    )
                                    if not cleaned_plain:
                                        st.session_state.conservation_edit_note_html = ""
                                        st.session_state.conservation_edit_note_plain = ""
                                        st.session_state[status_key] = "cleared"
                                        st.rerun()
                                    st.session_state.conservation_edit_note_html = cleaned_html
                                    st.session_state.conservation_edit_note_plain = cleaned_plain
                                    st.session_state[status_key] = "saved"
                                    st.rerun()

                        with st.form("conservation_edit_form", clear_on_submit=False):
                            col1, col2 = st.columns(2)
                            edit_action_value = col1.text_input(
                                "Action (583$a)",
                                key="conservation_edit_action",
                            )
                            edit_date_value = col2.text_input(
                                "Date (583$c)",
                                key="conservation_edit_date",
                            )
                            edit_status_value = st.text_input(
                                "Status (583$l)",
                                key="conservation_edit_status",
                            )
                            edit_agent_value = st.text_input(
                                "Agent(s) (583$k)",
                                help="Separate multiple names with semicolons.",
                                key="conservation_edit_agents",
                            )
                            edit_method_value = st.text_input(
                                "Method (583$i)",
                                key="conservation_edit_method",
                            )
                            edit_materials_value = st.text_input(
                                "Materials (583$3)",
                                key="conservation_edit_materials",
                            )
                            edit_extent_value = st.text_input(
                                "Extent (583$n)",
                                key="conservation_edit_extent",
                            )
                            edit_institution_value = st.text_input(
                                "Institution (583$5)",
                                key="conservation_edit_institution",
                            )
                            edit_submitted = st.form_submit_button("Save Action Changes")

                        if edit_submitted and edit_node:
                            if not edit_action_value.strip():
                                st.warning("Action is required.")
                            else:
                                edit_action_data: Dict[str, Any] = {}
                                edit_action_data["id"] = edit_node.id
                                edit_action_data["action"] = edit_action_value.strip()
                                if edit_date_value.strip():
                                    edit_action_data["date"] = edit_date_value.strip()
                                if edit_status_value.strip():
                                    edit_action_data["status"] = edit_status_value.strip()
                                if edit_method_value.strip():
                                    edit_action_data["method"] = edit_method_value.strip()
                                if edit_materials_value.strip():
                                    edit_action_data["materials"] = edit_materials_value.strip()
                                if edit_extent_value.strip():
                                    edit_action_data["extent"] = edit_extent_value.strip()
                                if edit_institution_value.strip():
                                    edit_action_data["institution"] = edit_institution_value.strip()

                                edit_agents = _split_agents(edit_agent_value)
                                if edit_agents:
                                    edit_action_data["agent"] = (
                                        edit_agents if len(edit_agents) > 1 else edit_agents[0]
                                    )

                                edit_note_plain = st.session_state.get(
                                    "conservation_edit_note_plain", ""
                                )
                                edit_note_html = st.session_state.get(
                                    "conservation_edit_note_html", ""
                                )
                                if edit_note_plain:
                                    edit_action_data["notes"] = [
                                        {"classified_as": "Annotation", "content": edit_note_plain}
                                    ]

                                detail_parts = []
                                if edit_date_value.strip():
                                    detail_parts.append(edit_date_value.strip())
                                if edit_status_value.strip():
                                    detail_parts.append(edit_status_value.strip())
                                edit_label = (
                                    f"{edit_action_value.strip()} ({'; '.join(detail_parts)})"
                                    if detail_parts
                                    else edit_action_value.strip()
                                )

                                edit_node.label = edit_label
                                edit_node.metadata.setdefault("source", "Manual")
                                edit_node.metadata["prefLabel"] = {"en": edit_label}
                                edit_node.metadata.update(edit_action_data)
                                if edit_note_plain:
                                    edit_node.metadata["annotation"] = edit_note_plain
                                else:
                                    edit_node.metadata.pop("annotation", None)
                                if edit_note_html:
                                    edit_node.metadata["annotation_html"] = edit_note_html
                                else:
                                    edit_node.metadata.pop("annotation_html", None)
                                st.session_state.id_to_label[edit_node.id] = edit_label

                                if work_node and isinstance(work_node.metadata, dict):
                                    existing_actions = work_node.metadata.get("583")
                                    if not isinstance(existing_actions, list):
                                        existing_actions = []
                                    replaced = False
                                    for idx, action_item in enumerate(existing_actions):
                                        if isinstance(action_item, dict) and action_item.get("id") == edit_node.id:
                                            existing_actions[idx] = edit_action_data
                                            replaced = True
                                            break
                                    if not replaced:
                                        existing_actions.append(edit_action_data)
                                    work_node.metadata["583"] = existing_actions

                                person_index: Dict[str, str] = {}
                                for node in st.session_state.graph_data.nodes:
                                    if "Person" not in [canonical_type(t) for t in (node.types or [])]:
                                        continue
                                    label = st.session_state.id_to_label.get(
                                        node.id, node.label or node.id
                                    )
                                    key = _normalize_label(label)
                                    if key and key not in person_index:
                                        person_index[key] = node.id

                                edit_node.edges = [
                                    edge for edge in edit_node.edges if edge.relationship != "performedBy"
                                ]
                                for agent in edit_agents:
                                    norm = _normalize_label(agent)
                                    if not norm:
                                        continue
                                    person_id = person_index.get(norm)
                                    if not person_id:
                                        person_id = f"urn:conservation:person:{uuid4().hex[:12]}"
                                        st.session_state.graph_data.nodes.append(
                                            Node(
                                                id=person_id,
                                                label=agent,
                                                types=["Person"],
                                                metadata={
                                                    "id": person_id,
                                                    "prefLabel": {"en": agent},
                                                    "type": ["Person"],
                                                    "source": "Manual",
                                                },
                                                edges=[],
                                            )
                                        )
                                        st.session_state.id_to_label[person_id] = agent
                                        person_index[norm] = person_id
                                    edit_node.edges.append(
                                        Edge(
                                            source=edit_node.id,
                                            target=person_id,
                                            relationship="performedBy",
                                        )
                                    )

                                st.success("Action updated.")
                                st.rerun()

                st.markdown("---")
                st.subheader("Add Conservation Action")

                st.markdown("#### Notes Annotation")
                status = st.session_state.pop("conservation_note_status", None)
                if status == "saved":
                    st.success("Notes saved.")
                elif status == "cleared":
                    st.info("Notes cleared.")
                note_html = st.session_state.get("conservation_note_html", "")
                note_plain = st.session_state.get("conservation_note_plain", "")
                note_event = _render_annotation_editor(
                    note_html,
                    "Add conservation notes (public or internal).",
                    key="conservation_note_editor",
                )
                if isinstance(note_event, dict) and note_event:
                    nonce_key = "conservation_note_nonce"
                    if note_event.get("nonce") != st.session_state.get(nonce_key):
                        st.session_state[nonce_key] = note_event.get("nonce")
                        if note_event.get("action") == "clear":
                            st.session_state.conservation_note_html = ""
                            st.session_state.conservation_note_plain = ""
                            st.session_state.conservation_note_status = "cleared"
                            st.rerun()
                        elif note_event.get("action") == "save":
                            cleaned_plain, cleaned_html = _resolve_annotation_payload(
                                note_event.get("html"),
                                note_event.get("text"),
                            )
                            if not cleaned_plain:
                                st.session_state.conservation_note_html = ""
                                st.session_state.conservation_note_plain = ""
                                st.session_state.conservation_note_status = "cleared"
                                st.rerun()
                            st.session_state.conservation_note_html = cleaned_html
                            st.session_state.conservation_note_plain = cleaned_plain
                            st.session_state.conservation_note_status = "saved"
                            st.rerun()

                note_html = st.session_state.get("conservation_note_html", "")
                note_plain = st.session_state.get("conservation_note_plain", "")
                if note_html:
                    safe_annotation = _sanitize_annotation_html(note_html)
                    st.markdown(
                        "<div class='annotation-preview'>"
                        f"{safe_annotation}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif note_plain:
                    st.info(note_plain)
                else:
                    st.caption("No notes saved yet.")

                with st.form("conservation_action_form", clear_on_submit=True):
                    col_a, col_b = st.columns(2)
                    action_value = col_a.text_input("Action (583$a)", key="conservation_action")
                    date_value = col_b.text_input("Date (583$c)", key="conservation_date")
                    status_value = st.text_input("Status (583$l)", key="conservation_status")
                    agent_value = st.text_input(
                        "Agent(s) (583$k)",
                        help="Separate multiple names with semicolons.",
                        key="conservation_agents",
                    )
                    method_value = st.text_input("Method (583$i)", key="conservation_method")
                    materials_value = st.text_input("Materials (583$3)", key="conservation_materials")
                    extent_value = st.text_input("Extent (583$n)", key="conservation_extent")
                    institution_value = st.text_input("Institution (583$5)", key="conservation_institution")
                    submitted = st.form_submit_button("Add Conservation Action")

                if submitted:
                    if not selected_work:
                        st.warning("Select a work before adding an action.")
                    elif not action_value.strip():
                        st.warning("Action is required.")
                    else:
                        action_data: Dict[str, Any] = {}
                        if action_value.strip():
                            action_data["action"] = action_value.strip()
                        if date_value.strip():
                            action_data["date"] = date_value.strip()
                        if status_value.strip():
                            action_data["status"] = status_value.strip()
                        if method_value.strip():
                            action_data["method"] = method_value.strip()
                        if materials_value.strip():
                            action_data["materials"] = materials_value.strip()
                        if extent_value.strip():
                            action_data["extent"] = extent_value.strip()
                        if institution_value.strip():
                            action_data["institution"] = institution_value.strip()

                        agents = _split_agents(agent_value)
                        if agents:
                            action_data["agent"] = agents if len(agents) > 1 else agents[0]

                        note_plain = st.session_state.get("conservation_note_plain", "")
                        note_html = st.session_state.get("conservation_note_html", "")
                        if not note_plain and note_html:
                            note_plain = _annotation_html_to_text(note_html)
                        if note_plain and not note_html:
                            note_html = _annotation_text_to_html(note_plain)
                        if note_plain:
                            action_data["notes"] = [{"classified_as": "Annotation", "content": note_plain}]

                        title = action_value.strip() or "Conservation action"
                        detail_parts = []
                        if date_value.strip():
                            detail_parts.append(date_value.strip())
                        if status_value.strip():
                            detail_parts.append(status_value.strip())
                        action_label = (
                            f"{title} ({'; '.join(detail_parts)})" if detail_parts else title
                        )
                        action_id = f"urn:conservation:{uuid4().hex[:12]}"

                        action_metadata = {
                            "id": action_id,
                            "prefLabel": {"en": action_label},
                            "type": ["ConservationAction"],
                            "source": "Manual",
                        }
                        action_metadata.update(action_data)
                        action_data["id"] = action_id
                        if note_plain:
                            action_metadata["annotation"] = note_plain
                        if note_html:
                            action_metadata["annotation_html"] = note_html

                        node_map = {n.id: n for n in st.session_state.graph_data.nodes}
                        if action_id not in node_map:
                            st.session_state.graph_data.nodes.append(
                                Node(
                                    id=action_id,
                                    label=action_label,
                                    types=["ConservationAction"],
                                    metadata=action_metadata,
                                    edges=[],
                                )
                            )
                        st.session_state.id_to_label[action_id] = action_label

                        if work_node:
                            work_node.edges.append(
                                Edge(
                                    source=work_node.id,
                                    target=action_id,
                                    relationship="conservationAction",
                                )
                            )
                            if isinstance(work_node.metadata, dict):
                                existing_actions = work_node.metadata.get("583")
                                if not isinstance(existing_actions, list):
                                    existing_actions = []
                                existing_actions.append(action_data)
                                work_node.metadata["583"] = existing_actions

                        person_index: Dict[str, str] = {}
                        for node in st.session_state.graph_data.nodes:
                            if "Person" not in [canonical_type(t) for t in (node.types or [])]:
                                continue
                            label = st.session_state.id_to_label.get(node.id, node.label or node.id)
                            key = _normalize_label(label)
                            if key and key not in person_index:
                                person_index[key] = node.id

                        for agent in agents:
                            norm = _normalize_label(agent)
                            if not norm:
                                continue
                            person_id = person_index.get(norm)
                            if not person_id:
                                person_id = f"urn:conservation:person:{uuid4().hex[:12]}"
                                st.session_state.graph_data.nodes.append(
                                    Node(
                                        id=person_id,
                                        label=agent,
                                        types=["Person"],
                                        metadata={
                                            "id": person_id,
                                            "prefLabel": {"en": agent},
                                            "type": ["Person"],
                                            "source": "Manual",
                                        },
                                        edges=[],
                                    )
                                )
                                st.session_state.id_to_label[person_id] = agent
                                person_index[norm] = person_id
                            action_node = next(
                                (n for n in st.session_state.graph_data.nodes if n.id == action_id),
                                None,
                            )
                            if action_node:
                                action_node.edges.append(
                                    Edge(
                                        source=action_id,
                                        target=person_id,
                                        relationship="performedBy",
                                    )
                                )

                        st.session_state.centrality_measures = None
                        st.session_state.graph_embeddings = None
                        st.session_state.textual_semantic_embeddings = None
                        st.session_state.node_roles = None
                        st.session_state.role_signature = None
                        st.session_state.node_anomalies = None
                        st.session_state.anomaly_signature = None

                        st.session_state.conservation_note_html = ""
                        st.session_state.conservation_note_plain = ""
                        st.success("Conservation action added.")
                        st.rerun()

            st.markdown("---")
            st.subheader("Conservation Graph")
            conservation_relationships = ["conservationAction", "performedBy"]
            candidate_nodes: Set[str] = set()
            for node in st.session_state.graph_data.nodes:
                if "ConservationAction" in [canonical_type(t) for t in (node.types or [])]:
                    candidate_nodes.add(node.id)
                for edge in node.edges:
                    if edge.relationship in conservation_relationships:
                        candidate_nodes.add(edge.source)
                        candidate_nodes.add(edge.target)
            if not candidate_nodes:
                st.caption("No conservation actions to display yet.")
            else:
                capped_nodes, cap_stats = apply_render_cap(
                    st.session_state.graph_data,
                    candidate_nodes,
                    MAX_RENDER_NODES,
                    conservation_relationships,
                    [],
                )
                if cap_stats["trimmed"] > 0:
                    st.info(
                        f"Render cap applied: showing {len(capped_nodes)} of {cap_stats['total']} nodes."
                    )
                graph_version = hash(
                    (
                        len(candidate_nodes),
                        len(st.session_state.graph_data.nodes),
                        tuple(conservation_relationships),
                    )
                )
                node_type_colors = st.session_state.get("node_type_colors") or CONFIG["NODE_TYPE_COLORS"]
                relationship_colors = st.session_state.get("relationship_colors") or CONFIG[
                    "RELATIONSHIP_CONFIG"
                ]
                graph_background = st.session_state.get("graph_background") or CONFIG.get(
                    "GRAPH_BACKGROUND", {}
                )
                community_colors = st.session_state.get("community_colors") or CONFIG.get(
                    "COMMUNITY_COLORS", []
                )
                community_locks = st.session_state.get("community_locks") or {}
                net = build_graph(
                    graph_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=conservation_relationships,
                    node_positions=st.session_state.node_positions,
                    show_labels=st.session_state.show_labels,
                    smart_labels=st.session_state.smart_labels,
                    label_zoom_threshold=st.session_state.label_zoom_threshold,
                    filtered_nodes=capped_nodes,
                    community_detection=False,
                    centrality=None,
                    path_nodes=None,
                    graph_version=graph_version,
                    reduce_motion=st.session_state.reduce_motion,
                    motion_intensity=st.session_state.motion_intensity,
                    node_animations=effective_node_animations,
                    node_animation_strength=st.session_state.node_animation_strength,
                    node_type_colors=node_type_colors,
                    relationship_colors=relationship_colors,
                    graph_background=graph_background,
                    community_colors=community_colors,
                    community_locks=community_locks,
                    focus_context=st.session_state.focus_context,
                    edge_semantics=st.session_state.edge_semantics,
                    type_icons=st.session_state.type_icons,
                )
                try:
                    html_code = net.html
                    components.html(html_code, height=GRAPH_CARD_HEIGHT, scrolling=False)
                except Exception as exc:
                    st.error(f"Conservation graph generation failed: {exc}")

    with tabs[2]:
        with st.expander("Data View", expanded=False):
            st.header("Data View")
            st.subheader("Graph Nodes")
            if st.session_state.graph_data.nodes:
                data_rows = []
                for n in st.session_state.graph_data.nodes:
                    safe_id = re.sub(r"[^\x20-\x7E]+", "", n.id)
                    safe_label = re.sub(r"[^\x20-\x7E]+", "", n.label)
                    canon_types = [canonical_type(t) for t in n.types]
                    safe_types = re.sub(r"[^\x20-\x7E]+", "", ", ".join(canon_types))
                    inferred_types = _extract_inferred_types(n.metadata)
                    inferred_display = "; ".join(_shorten_iri(t) for t in inferred_types)
                    data_rows.append(
                        {
                            "ID": safe_id,
                            "Label": safe_label,
                            "Types": safe_types,
                            "Inferred Type Count": len(inferred_types),
                            "Inferred Types": inferred_display,
                        }
                    )

                df_nodes = pd.DataFrame(data_rows)
                st.dataframe(df_nodes, use_container_width=True)
                csv_data = df_nodes.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Nodes as CSV", data=csv_data, file_name="nodes.csv", mime="text/csv"
                )
            else:
                st.info("No data available. Please upload JSON/RDF files.")

    with tabs[3]:
        with st.expander("Centrality Measures", expanded=False):
            st.header("Centrality Measures")
            if st.session_state.centrality_measures:
                centrality_df = pd.DataFrame.from_dict(
                    st.session_state.centrality_measures, orient="index"
                ).reset_index().rename(columns={"index": "Node ID"})
                centrality_df["Label"] = centrality_df["Node ID"].map(st.session_state.id_to_label)
                id_to_type = {}
                for n in st.session_state.graph_data.nodes:
                    chosen_type = "Unknown"
                    if n.types:
                        for t in n.types:
                            canon = canonical_type(t)
                            if canon != "Unknown":
                                chosen_type = canon
                                break
                    id_to_type[n.id] = chosen_type
                centrality_df["Type"] = centrality_df["Node ID"].map(id_to_type).fillna("Unknown")

                cols = ["Node ID", "Label", "Type"] + [
                    col for col in centrality_df.columns if col not in ("Node ID", "Label", "Type")
                ]
                centrality_df = centrality_df[cols]
                st.dataframe(centrality_df, use_container_width=True)

                metrics = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]

                st.subheader("Top-K Lollipop Panels")
                hide_unnamed = st.checkbox("Hide unnamed / URI labels", value=True)
                lollipop_df = centrality_df.copy()
                if hide_unnamed:
                    lollipop_df = lollipop_df[
                        ~lollipop_df.apply(
                            lambda row: _label_is_unnamed(row["Label"], row["Node ID"]), axis=1
                        )
                    ]

                if lollipop_df.empty:
                    st.info("No named nodes available for Top-K lollipop panels.")
                else:
                    k_max = max(1, min(25, len(lollipop_df)))
                    k_default = min(10, k_max)
                    top_k = st.slider("Top K nodes per metric", min_value=1, max_value=k_max, value=k_default)
                    columns = st.columns(2)
                    for idx, metric in enumerate(metrics):
                        top = lollipop_df.sort_values(metric, ascending=False).head(top_k).copy()
                        top["Display"] = top["Label"].fillna(top["Node ID"])
                        top = top.iloc[::-1]
                        point_colors = top["Type"].map(CONFIG["NODE_TYPE_COLORS"]).fillna(
                            CONFIG["DEFAULT_NODE_COLOR"]
                        )

                        fig_lollipop = go.Figure()
                        for y_label, val in zip(top["Display"], top[metric]):
                            fig_lollipop.add_trace(
                                go.Scatter(
                                    x=[0, val],
                                    y=[y_label, y_label],
                                    mode="lines",
                                    line=dict(color="#CBD5E1", width=2),
                                    hoverinfo="skip",
                                    showlegend=False,
                                )
                            )
                        fig_lollipop.add_trace(
                            go.Scatter(
                                x=top[metric],
                                y=top["Display"],
                                mode="markers",
                                marker=dict(size=10, color=point_colors, line=dict(color="#1F2937", width=0.6)),
                                customdata=np.stack([top["Node ID"], top["Type"]], axis=-1),
                                hovertemplate="<b>%{y}</b><br>Value: %{x:.4f}<br>ID: %{customdata[0]}<br>Type: %{customdata[1]}<extra></extra>",
                                showlegend=False,
                            )
                        )
                        fig_lollipop.update_layout(
                            title=f"Top {top_k} - {metric.title()}",
                            height=360,
                            margin=dict(l=30, r=20, t=60, b=30),
                            xaxis_title=metric.title(),
                            yaxis_title="",
                            template="plotly_white",
                        )
                        fig_lollipop.update_yaxes(automargin=True)
                        columns[idx % 2].plotly_chart(fig_lollipop, use_container_width=True)

                st.subheader("Rank-Flow (Bump Chart)")
                bump_max = max(3, min(20, len(centrality_df)))
                bump_default = min(12, bump_max)
                bump_top = st.slider(
                    "Top nodes for rank-flow", min_value=3, max_value=bump_max, value=bump_default
                )
                rank_df = centrality_df.copy()
                for metric in metrics:
                    rank_df[f"{metric}_rank"] = rank_df[metric].rank(ascending=False, method="min")
                rank_df["Mean Rank"] = rank_df[[f"{m}_rank" for m in metrics]].mean(axis=1)
                bump_nodes = rank_df.nsmallest(bump_top, "Mean Rank").copy()

                bump_frames = []
                for metric in metrics:
                    tmp = bump_nodes[["Node ID", "Label", "Type", f"{metric}_rank"]].copy()
                    tmp["Metric"] = metric.title()
                    tmp = tmp.rename(columns={f"{metric}_rank": "Rank"})
                    bump_frames.append(tmp)
                bump_plot_df = pd.concat(bump_frames, ignore_index=True)

                fig_bump = px.line(
                    bump_plot_df,
                    x="Metric",
                    y="Rank",
                    color="Label",
                    line_group="Node ID",
                    markers=True,
                    hover_data=["Node ID", "Type", "Rank"],
                    title="Rank-Flow Across Centrality Metrics",
                )
                fig_bump.update_layout(
                    height=420,
                    margin=dict(l=30, r=30, t=60, b=40),
                    xaxis_title="Metric",
                    yaxis_title="Rank (1 = highest)",
                )
                fig_bump.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_bump, use_container_width=True)

                csv_data = centrality_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Centrality Measures as CSV",
                    data=csv_data,
                    file_name="centrality_measures.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "Centrality measures have not been computed yet. Please enable 'Display Centrality Measures'."
                )

    with tabs[4]:
        with st.expander("SPARQL Query", expanded=False):
            st.header("SPARQL Query")
            st.markdown("Enter a SPARQL SELECT query in the sidebar and view the results here.")
            if st.session_state.sparql_query.strip():
                try:
                    query_results = run_sparql_query(st.session_state.sparql_query, st.session_state.rdf_graph)
                    st.success(f"Query returned {len(query_results)} result(s).")
                    st.dataframe(
                        pd.DataFrame(list(query_results), columns=["Node ID"]), use_container_width=True
                    )
                except Exception as exc:
                    st.error(f"SPARQL Query failed: {exc}")
            else:
                st.info("No query entered.")
            st.markdown("---")
            st.subheader("Run against OCLC Entity Query (remote)")
            oclc_query = st.text_area(
                "OCLC SPARQL",
                value="",
                key="oclc_remote_q",
                placeholder="Enter a SPARQL query (SELECT/CONSTRUCT/DESCRIBE)",
            )
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                if st.button("Run SELECT (remote)"):
                    if not oclc_query.strip():
                        st.warning("Enter a SPARQL SELECT query first.")
                    else:
                        try:
                            res = oclc_sparql(oclc_query, result_type="select")
                            vars_ = res.get("head", {}).get("vars", [])
                            rows = res.get("results", {}).get("bindings", [])
                            if rows:
                                df_rows = [{k: b.get(k, {}).get("value", "") for k in vars_} for b in rows]
                                st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
                            else:
                                st.json(res)
                        except Exception as exc:
                            st.error(f"Remote SELECT failed: {exc}")

            with c2:
                if st.button("Run CONSTRUCT -> Load Graph"):
                    if not oclc_query.strip():
                        st.warning("Enter a SPARQL CONSTRUCT/DESCRIBE query first.")
                    else:
                        try:
                            if oclc_query.strip().upper().startswith("SELECT"):
                                st.info("Tip: Change this to a CONSTRUCT/DESCRIBE query to build a graph.")
                            res = oclc_sparql(oclc_query, result_type="graph")
                            graph_data, id_to_label = load_jsonld_into_session(res)
                            st.session_state.graph_data = graph_data
                            st.session_state.id_to_label = id_to_label
                            try:
                                st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
                            except Exception as exc:
                                st.warning(f"Graph converted with a warning: {exc}")
                            st.success(f"Loaded {len(graph_data.nodes)} node(s) from JSON-LD into the graph.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"CONSTRUCT/Load failed: {exc}")

            with c3:
                st.caption("Credentials read from secrets/env: OCLC_CLIENT_ID / OCLC_CLIENT_SECRET")

    with tabs[5]:
        with st.expander("Timeline View", expanded=False):
            st.header("Timeline View")
            timeline_data = []

            for n in st.session_state.graph_data.nodes:
                dob = n.metadata.get("dateOfBirth")
                if isinstance(dob, list) and dob:
                    dval = dob[0].get("time:inXSDDateTimeStamp", {}).get("@value")
                    if dval:
                        timeline_data.append({"Label": n.label, "Event": "Birth", "Date": dval})

                dod = n.metadata.get("dateOfDeath")
                if isinstance(dod, list) and dod:
                    dval = dod[0].get("time:inXSDDateTimeStamp", {}).get("@value")
                    if dval:
                        timeline_data.append({"Label": n.label, "Event": "Death", "Date": dval})

                for rel in ["educatedAt", "employedBy"]:
                    events = n.metadata.get(rel)
                    if events:
                        if not isinstance(events, list):
                            events = [events]
                        for ev in events:
                            if isinstance(ev, dict):
                                start = ev.get("startDate")
                                if start:
                                    val_start = start.get("time:inXSDDateTimeStamp", {}).get("@value")
                                    if val_start:
                                        timeline_data.append(
                                            {"Label": n.label, "Event": f"{rel} Start", "Date": val_start}
                                        )
                                end = ev.get("endDate")
                                if end:
                                    val_end = end.get("time:inXSDDateTimeStamp", {}).get("@value")
                                    if val_end:
                                        timeline_data.append(
                                            {"Label": n.label, "Event": f"{rel} End", "Date": val_end}
                                        )

            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                df_timeline["Date"] = df_timeline["Date"].apply(lambda x: parse_date(x))
                fig_static = px.scatter(
                    df_timeline,
                    x="Date",
                    y="Label",
                    color="Event",
                    hover_data=["Event", "Date"],
                    title="Entity Timeline (Scatter Plot)",
                )
                fig_static.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_static, use_container_width=True)
            else:
                st.info("No timeline data available.")

    with tabs[6]:
        with st.expander("Graph Embeddings", expanded=False):
            st.header("Graph Embeddings")
            if st.session_state.graph_embeddings:
                emb = st.session_state.graph_embeddings
                emb_data = []
                for node, vec in emb.items():
                    emb_data.append({"Node": node, "Embedding": ", ".join(f"{x:.3f}" for x in vec[:5]) + " ..."})
                df_emb = pd.DataFrame(emb_data)
                st.dataframe(df_emb, use_container_width=True)
                csv_data = df_emb.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Embeddings as CSV",
                    data=csv_data,
                    file_name="graph_embeddings.csv",
                    mime="text/csv",
                )
            else:
                st.info("Graph embeddings not computed yet. Use the sidebar to compute them.")

    with tabs[7]:
        with st.expander("Node Similarity Search", expanded=False):
            st.header("Node Similarity Search")
            similarity_type = st.selectbox("Select Similarity Type", ["Graph Embedding", "Textual Semantic"])

            if similarity_type == "Graph Embedding":
                if st.session_state.graph_embeddings:
                    embeddings = st.session_state.graph_embeddings
                else:
                    st.info("Please compute the graph embeddings first using the sidebar!")
                    embeddings = None
            else:
                if not sentence_transformer_installed:
                    st.error(
                        "SentenceTransformer is not installed. Install it to use Textual Semantic Similarity analysis."
                    )
                    embeddings = None
                else:
                    if st.session_state.textual_semantic_embeddings is None:
                        if st.button("Compute Textual Semantic Embeddings"):
                            embeddings = compute_textual_semantic_embeddings(st.session_state.graph_data)
                            st.session_state.textual_semantic_embeddings = embeddings
                            st.success("Textual Semantic Embeddings computed!")
                        else:
                            embeddings = None
                    else:
                        embeddings = st.session_state.textual_semantic_embeddings

            if embeddings:
                node_list = list(embeddings.keys())
                selected_node = st.selectbox(
                    "Select a node to find similar nodes",
                    node_list,
                    format_func=lambda x: st.session_state.id_to_label.get(x, x),
                )
                if selected_node:
                    selected_vector = np.array(embeddings[selected_node])
                    similarities = {}
                    for node, vector in embeddings.items():
                        if node == selected_node:
                            continue
                        vec = np.array(vector)
                        denom = np.linalg.norm(selected_vector) * np.linalg.norm(vec)
                        sim = float(np.dot(selected_vector, vec) / denom) if denom != 0 else 0.0
                        similarities[node] = sim

                    sorted_sim = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    df_sim = pd.DataFrame(sorted_sim, columns=["Node", "Cosine Similarity"]).head(10)
                    df_sim["Node Label"] = df_sim["Node"].map(
                        lambda x: st.session_state.id_to_label.get(x, x)
                    )
                    st.dataframe(df_sim, use_container_width=True)

                    st.subheader("Similarity Constellation")
                    if not umap_installed:
                        st.error("UMAP is not installed. Add `umap-learn` to compute the constellation.")
                    elif not sorted_sim:
                        st.info("No neighbors available for constellation view.")
                    else:
                        max_nodes = min(50, len(sorted_sim))
                        min_nodes = 5 if max_nodes >= 5 else max_nodes
                        default_nodes = min(20, max_nodes) if max_nodes >= 5 else max_nodes
                        top_k = st.slider(
                            "Constellation size",
                            min_value=min_nodes,
                            max_value=max_nodes,
                            value=default_nodes,
                            step=1,
                        )
                        nodes_for_plot = [selected_node] + [node for node, _ in sorted_sim[:top_k]]
                        sims_for_plot = [1.0] + [sim for _, sim in sorted_sim[:top_k]]

                        if len(nodes_for_plot) < 3:
                            st.info("Add more nodes to build a UMAP constellation.")
                        else:
                            labels = [st.session_state.id_to_label.get(nid, nid) for nid in nodes_for_plot]
                            base_color = "#2D4F6A"
                            light_target = "#F4EFE7"
                            colors = []
                            for idx, sim_val in enumerate(sims_for_plot):
                                sim_clamped = max(0.0, min(1.0, float(sim_val)))
                                if idx == 0:
                                    ratio = 0.0
                                else:
                                    ratio = 0.75 * (1.0 - sim_clamped)
                                colors.append(_blend_hex(base_color, light_target, ratio))

                            vectors = np.array([embeddings[nid] for nid in nodes_for_plot], dtype=float)
                            try:
                                n_neighbors = min(10, len(nodes_for_plot) - 1)
                                if n_neighbors < 2:
                                    n_neighbors = 2
                                reducer = umap.UMAP(
                                    n_components=2,
                                    n_neighbors=n_neighbors,
                                    min_dist=0.2,
                                    metric="cosine",
                                    random_state=42,
                                )
                                coords = reducer.fit_transform(vectors)
                            except Exception as exc:
                                st.error(f"UMAP failed to compute the constellation: {exc}")
                            else:
                                coords = coords - coords[0]
                                radii = np.linalg.norm(coords, axis=1)
                                max_r = float(np.max(radii[1:])) if len(radii) > 1 else 1.0
                                if max_r == 0:
                                    max_r = 1.0
                                min_radius = max_r * 0.04

                                adjusted = coords.copy()
                                for idx in range(1, len(nodes_for_plot)):
                                    r = radii[idx]
                                    sim = max(0.0, min(1.0, float(sims_for_plot[idx])))
                                    if sim >= 0.999:
                                        target_r = max_r * 0.08
                                    else:
                                        target_r = (1.0 - sim) * max_r
                                    if target_r < min_radius:
                                        target_r = min_radius
                                    if r > 0:
                                        adjusted[idx] = coords[idx] * (target_r / r)
                                    else:
                                        angle = (hash(nodes_for_plot[idx]) % 360) * (np.pi / 180.0)
                                        adjusted[idx] = np.array([np.cos(angle), np.sin(angle)]) * target_r
                                adjusted[0] = np.array([0.0, 0.0])

                                sizes = []
                                aura_sizes = []
                                hover = []
                                halo_scale = 2.3
                                halo_alpha = 0.18
                                for idx, nid in enumerate(nodes_for_plot):
                                    sim_val = 1.0 if idx == 0 else sims_for_plot[idx]
                                    if idx == 0:
                                        size = 30
                                        aura = size * halo_scale * 1.1
                                    else:
                                        sim_clamped = max(0.0, min(1.0, float(sim_val)))
                                        size = 14 + sim_clamped * 18
                                        aura = size * halo_scale
                                    sizes.append(size)
                                    aura_sizes.append(aura)
                                    hover.append(
                                        f"{labels[idx]}<br>Similarity: {sim_val:.3f}<br>ID: {nid}"
                                    )

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=adjusted[:, 0],
                                        y=adjusted[:, 1],
                                        mode="markers",
                                        marker=dict(size=aura_sizes, color=colors, opacity=halo_alpha),
                                        hoverinfo="skip",
                                        showlegend=False,
                                    )
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=adjusted[:, 0],
                                        y=adjusted[:, 1],
                                        mode="markers+text",
                                        text=labels,
                                        textposition="top center",
                                        marker=dict(
                                            size=sizes,
                                            color=colors,
                                            line=dict(color="rgba(15, 23, 42, 0.45)", width=1.2),
                                        ),
                                        textfont=dict(size=9, color="#1F2A37"),
                                        hovertext=hover,
                                        hoverinfo="text",
                                        showlegend=False,
                                    )
                                )

                                ring_thresholds = [0.2, 0.4, 0.6, 0.8]
                                rings = []
                                for threshold in ring_thresholds:
                                    r = (1.0 - threshold) * max_r
                                    rings.append(
                                        dict(
                                            type="circle",
                                            xref="x",
                                            yref="y",
                                            x0=-r,
                                            y0=-r,
                                            x1=r,
                                            y1=r,
                                            line=dict(color="rgba(15, 23, 42, 0.15)", width=1, dash="dot"),
                                            fillcolor="rgba(0, 0, 0, 0)",
                                        )
                                    )

                                fig.update_layout(
                                    height=520,
                                    margin=dict(l=20, r=20, t=30, b=20),
                                    paper_bgcolor="rgba(248, 246, 241, 0.92)",
                                    plot_bgcolor="rgba(248, 246, 241, 0.0)",
                                    xaxis=dict(visible=False, showgrid=False, zeroline=False),
                                    yaxis=dict(
                                        visible=False,
                                        showgrid=False,
                                        zeroline=False,
                                        scaleanchor="x",
                                        scaleratio=1,
                                    ),
                                    shapes=rings,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("Radial rings indicate similarity bands (0.8, 0.6, 0.4, 0.2).")
                else:
                    st.info("Select a node to search for similar nodes.")
            else:
                st.info("No embeddings available for similarity search. Please compute the required embeddings above!")

    with tabs[8]:
        with st.expander("About", expanded=False):
            st.header("About Linked Data Explorer")
            st.markdown(
                """
                ### Explore Relationships Between Entities
                Upload multiple JSON or RDF files representing entities and generate an interactive network.
                Use the sidebar to filter relationships, search for nodes, set manual positions,
                and edit the graph directly.

                **Enhanced Features:**
                - **RDF Ingestion & SHACL Validation:** Directly load RDF (Turtle, RDF/XML, N-Triples) and validate with SHACL.
                - **URI Dereferencing:** Fetch and integrate external RDF data.
                - **RDFS Reasoning:** Infer implicit relationships using basic RDFS reasoning.
                - **Ontology Suggestions:** Get ontology suggestions based on your data.
                - **Advanced SPARQL Querying:** Run complex queries.
                - **Semantic Graph Visualization:** Nodes and edges are styled based on RDF types and properties.
                - **Graph Embeddings:** Integrated probabilistic graph embedding model (via node2vec) for learning latent node representations.
                - **Textual Semantic Analysis:** Compute text-based semantic embeddings for nodes using SentenceTransformer.
                - **Node Similarity Search:** Find similar nodes based on cosine similarity of embeddings (graph or textual).
                - **Pathfinding, Node Annotations, and Graph Editing:** Interactive tools for network exploration.

                **Version:** 2.1.4
                **Author:** Huw Sandaver (Refactored by ChatGPT)
                **Contact:** hsandaver@alumni.unimelb.edu.au

                Enjoy exploring your linked data!
                """
            )
