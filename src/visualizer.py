"""PyVis, PyDeck, Plotly generation helpers."""

from __future__ import annotations

import html as html_lib
import io
import json
import logging
import math
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import networkx as nx
import streamlit as st
from pyvis.network import Network

from src.config import (
    APP_FONTS,
    CONFIG,
    GRAPH_CANVAS_HEIGHT,
    GRAPH_CARD_HEIGHT,
    GRAPH_PANEL_PADDING,
    VIS_FONT_FACE,
)
from src.models import GraphData
from src.utils import (
    _extract_inferred_types,
    _hex_to_rgb,
    _rgb_to_hex,
    _local_name,
    _make_edge_color,
    _make_node_color,
    _pick_label_color,
    _render_metadata_overview,
    _shorten_iri,
    canonical_type,
    refresh_label_index,
)

try:
    import community.community_louvain as community_louvain

    louvain_installed = True
except ImportError:
    louvain_installed = False


_HEX_COLOR_RE = re.compile(r"^[0-9a-fA-F]{6}$")
_GRAPH_BG_DEFAULTS = {
    "radial_1": "#E07A5F",
    "radial_2": "#2A9D8F",
    "linear_1": "#FCFAF6",
    "linear_2": "#F4EFE6",
}
_COMMUNITY_DEFAULTS = [
    "#3D5A80",
    "#E07A5F",
    "#2A9D8F",
    "#F4A261",
    "#6D597A",
    "#84A59D",
    "#F2D388",
    "#B56576",
    "#7AA4C1",
    "#5C5F66",
]
_EXPORT_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_EXPORT_EVENT_RE = re.compile(r"\son\w+\s*=\s*(['\"]).*?\1", re.IGNORECASE | re.DOTALL)
_EXPORT_JS_URL_RE = re.compile(r"href\s*=\s*(['\"])\s*javascript:[^'\"]*\1", re.IGNORECASE)
_EXPORT_RESERVED_KEYS = {"id", "@id", "prefLabel", "type", "inferredTypes", "annotation", "annotation_html"}
_TOOLTIP_TAG_RE = re.compile(r"<[^>]+>")
_TOOLTIP_BREAK_RE = re.compile(r"<\s*(br\s*/?|/p\s*|/li\s*)>", re.IGNORECASE)


def _normalize_hex_color(value: Optional[str], fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if not _HEX_COLOR_RE.match(text):
        return fallback
    return f"#{text}"


def _normalize_hex_color_or_none(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if not _HEX_COLOR_RE.match(text):
        return None
    return f"#{text}"


def _coerce_color_list(values: Optional[List[str]], defaults: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(defaults)
    cleaned: List[str] = []
    for value in values:
        normalized = _normalize_hex_color_or_none(value)
        if normalized:
            cleaned.append(normalized)
    return cleaned if cleaned else list(defaults)


def _coerce_label_list(values: Optional[List[str]], defaults: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(defaults)
    cleaned: List[str] = []
    for value in values:
        if value is None:
            cleaned.append("")
        else:
            cleaned.append(str(value))
    return cleaned if cleaned else list(defaults)


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    hue = hue % 1.0
    saturation = max(0.0, min(1.0, saturation))
    lightness = max(0.0, min(1.0, lightness))

    if saturation == 0:
        channel = int(round(lightness * 255))
        return _rgb_to_hex((channel, channel, channel))

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
    return _rgb_to_hex((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))


def _ensure_palette_size(base_palette: List[str], target_count: int) -> List[str]:
    palette = list(base_palette)
    if target_count <= len(palette):
        return palette[:target_count]
    golden_ratio = 0.61803398875
    idx = len(palette)
    while len(palette) < target_count:
        hue = (idx * golden_ratio) % 1.0
        palette.append(_hsl_to_hex(hue, 0.55, 0.55))
        idx += 1
    return palette


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


_TYPE_ICON_CACHE: Dict[Tuple[str, str, str], str] = {}
_TYPE_ICON_SVGS: Dict[str, str] = {
    "Person": (
        "<circle cx='32' cy='22' r='9'/>"
        "<path d='M14 52c3-9 12-14 18-14s15 5 18 14'/>"
    ),
    "Organization": (
        "<rect x='16' y='20' width='32' height='28' rx='2'/>"
        "<path d='M22 26h6M22 32h6M22 38h6M36 26h6M36 32h6M36 38h6'/>"
        "<path d='M28 48v-8h8v8'/>"
    ),
    "Place": (
        "<path d='M32 12c-8 0-14 6-14 14 0 11 14 26 14 26s14-15 14-26c0-8-6-14-14-14z'/>"
        "<circle cx='32' cy='26' r='5'/>"
    ),
    "StillImage": (
        "<rect x='14' y='18' width='36' height='28' rx='2'/>"
        "<circle cx='24' cy='28' r='3' fill='{fg}' stroke='none'/>"
        "<path d='M20 42l8-8 6 6 6-7 8 9'/>"
    ),
    "Event": (
        "<rect x='16' y='20' width='32' height='28' rx='2'/>"
        "<path d='M16 28h32'/>"
        "<path d='M24 16v6M40 16v6'/>"
        "<path d='M24 34h6M34 34h6M24 40h6'/>"
    ),
    "Work": (
        "<path d='M20 14h18l8 8v28H20z'/>"
        "<path d='M38 14v8h8'/>"
        "<path d='M24 32h16M24 38h16'/>"
    ),
    "Music": (
        "<path d='M38 18v22'/>"
        "<circle cx='26' cy='40' r='6'/>"
        "<circle cx='38' cy='36' r='6'/>"
        "<path d='M38 18l12-3'/>"
    ),
    "AdministrativeArea": (
        "<path d='M16 18l12-4 12 4 8-4v32l-8 4-12-4-12 4-8-4V18z'/>"
        "<path d='M28 14v36M40 18v32'/>"
    ),
    "Unknown": (
        "<path d='M24 26c0-4 3-7 8-7s8 3 8 7c0 5-4 6-6 8-1 1-2 2-2 4'/>"
        "<circle cx='32' cy='44' r='2.5' fill='{fg}' stroke='none'/>"
    ),
}


def _type_icon_data(icon_key: str, base_color: str, fallback_text: str) -> str:
    key = (icon_key, base_color, fallback_text)
    cached = _TYPE_ICON_CACHE.get(key)
    if cached:
        return cached
    label_color = _pick_label_color(base_color)
    icon_svg = _TYPE_ICON_SVGS.get(icon_key, "")
    if icon_svg:
        icon_group = (
            "<g fill='none' stroke='{fg}' stroke-width='2.6' "
            "stroke-linecap='round' stroke-linejoin='round'>"
            f"{icon_svg}"
            "</g>"
        ).format(fg=label_color)
    else:
        glyph = html_lib.escape((fallback_text or "?")[:2])
        icon_group = (
            f"<text x='32' y='38' text-anchor='middle' font-size='26' "
            "font-family='IBM Plex Sans, Arial, sans-serif' font-weight='700' "
            f"fill='{label_color}'>{glyph}</text>"
        )
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
        f"<circle cx='32' cy='32' r='28' fill='{base_color}' "
        "stroke='rgba(15, 23, 42, 0.18)' stroke-width='2'/>"
        f"{icon_group}"
        "</svg>"
    )
    data_uri = "data:image/svg+xml;utf8," + quote(svg)
    _TYPE_ICON_CACHE[key] = data_uri
    return data_uri


def _edge_width_from_weight(weight: Optional[float], base_width: float) -> Optional[float]:
    if weight is None:
        return None
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return max(1.0, base_width * 0.75)
    if value <= 1:
        return base_width + (value * 2.4)
    return base_width + min(3.5, math.log1p(value) * 1.6)


def _build_graph_background(background: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    defaults = CONFIG.get("GRAPH_BACKGROUND") or _GRAPH_BG_DEFAULTS
    radial_1 = _normalize_hex_color(
        (background or {}).get("radial_1"), defaults.get("radial_1", _GRAPH_BG_DEFAULTS["radial_1"])
    )
    radial_2 = _normalize_hex_color(
        (background or {}).get("radial_2"), defaults.get("radial_2", _GRAPH_BG_DEFAULTS["radial_2"])
    )
    linear_1 = _normalize_hex_color(
        (background or {}).get("linear_1"), defaults.get("linear_1", _GRAPH_BG_DEFAULTS["linear_1"])
    )
    linear_2 = _normalize_hex_color(
        (background or {}).get("linear_2"), defaults.get("linear_2", _GRAPH_BG_DEFAULTS["linear_2"])
    )
    radial_1_rgba = _hex_to_rgba(radial_1, 0.18)
    radial_2_rgba = _hex_to_rgba(radial_2, 0.18)
    gradient = (
        "radial-gradient(circle at 15% 20%, "
        f"{radial_1_rgba}, transparent 45%), "
        "radial-gradient(circle at 80% 0%, "
        f"{radial_2_rgba}, transparent 40%), "
        f"linear-gradient(135deg, {linear_1} 0%, {linear_2} 100%)"
    )
    return gradient, linear_2


def _is_annotation_classification(value: Any) -> bool:
    if not value:
        return False
    if isinstance(value, str):
        return "annotation" in value.lower()
    if isinstance(value, list):
        return any("annotation" in str(item).lower() for item in value if item)
    return False


def _extract_annotation_from_notes(metadata: Dict[str, Any]) -> str:
    values = metadata.get("notes")
    if not values:
        return ""
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return ""
    for entry in values:
        if not isinstance(entry, dict):
            continue
        if not _is_annotation_classification(entry.get("classified_as") or entry.get("classification")):
            continue
        for key in ("content", "value", "@value", "_label", "label"):
            raw = entry.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return ""


def _extract_metadata_notes(metadata: Dict[str, Any], skip_annotation: bool = False) -> List[str]:
    def _collect(values: Any) -> List[str]:
        if not values:
            return []
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            return []
        notes: List[str] = []
        for entry in values:
            if not isinstance(entry, dict):
                continue
            if skip_annotation and _is_annotation_classification(
                entry.get("classified_as") or entry.get("classification")
            ):
                continue
            for key in ("content", "value", "@value", "_label", "label"):
                raw = entry.get(key)
                if isinstance(raw, str) and raw.strip():
                    notes.append(raw.strip())
                    break
        return notes

    notes: List[str] = []
    notes.extend(_collect(metadata.get("subject_of")))
    notes.extend(_collect(metadata.get("referred_to_by")))
    notes.extend(_collect(metadata.get("notes")))
    notes.extend(_collect(metadata.get("note")))
    seen = set()
    unique: List[str] = []
    for note in notes:
        key = note.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(note)
    return unique


def _clean_tooltip_text(value: str) -> str:
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFKC", value)
    cleaned = cleaned.replace("\u2028", "\n").replace("\u2029", "\n").replace("\u0085", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")
    cleaned = re.sub(r"(?<=[A-Za-z])\s*\n\s*(?=[A-Za-z])", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _annotation_html_to_text(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _TOOLTIP_BREAK_RE.sub("\n", value)
    cleaned = _EXPORT_SCRIPT_RE.sub("", cleaned)
    cleaned = _TOOLTIP_TAG_RE.sub("", cleaned)
    return html_lib.unescape(cleaned)


def _sanitize_export_html(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    cleaned = _EXPORT_SCRIPT_RE.sub("", cleaned)
    cleaned = _EXPORT_EVENT_RE.sub("", cleaned)
    cleaned = _EXPORT_JS_URL_RE.sub('href="#"', cleaned)
    return cleaned


def _build_export_metadata_map(
    graph_data: GraphData,
    label_lookup: Optional[Dict[str, str]] = None,
    max_cards: int = 12,
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    label_lookup = label_lookup or {}
    for node in graph_data.nodes:
        node_id = str(node.id)
        label = str(node.label) if node.label is not None and str(node.label).strip() else node_id
        types = [canonical_type(t) for t in (node.types or []) if t]
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        annotation_html = _sanitize_export_html(str(meta.get("annotation_html") or ""))
        annotation_text = str(meta.get("annotation") or "").strip()
        if not annotation_html and annotation_text:
            annotation_html = html_lib.escape(annotation_text).replace("\n", "<br>")
        metadata_html = _render_metadata_overview(
            meta,
            skip_keys=_EXPORT_RESERVED_KEYS,
            max_cards=max_cards,
            label_lookup=label_lookup,
        )
        metadata_html = _sanitize_export_html(metadata_html or "")
        properties: List[Dict[str, str]] = []
        payload[node_id] = {
            "id": node_id,
            "label": label,
            "types": types,
            "annotation_html": annotation_html,
            "annotation_text": html_lib.escape(annotation_text) if annotation_text else "",
            "metadata_html": metadata_html or "",
            "properties": properties,
        }
    return payload


def add_node(
    net: Network,
    node_id: str,
    label: str,
    entity_types: List[str],
    metadata: Dict[str, Any],
    search_nodes: Optional[List[str]] = None,
    show_labels: bool = True,
    smart_labels: bool = False,
    custom_size: Optional[int] = None,
    node_type_colors: Optional[Dict[str, str]] = None,
    type_icons: bool = False,
) -> None:
    canon_types = [canonical_type(t) for t in (entity_types or [])] or ["Unknown"]
    palette = node_type_colors or {}
    base_palette = CONFIG["NODE_TYPE_COLORS"]
    primary_type = next((t for t in canon_types if t in palette or t in base_palette), "Unknown")

    if any(isinstance(t, str) and "foaf" in t.lower() for t in (entity_types or [])):
        color = "#FFA07A"
        shape = "star"
    else:
        color = palette.get(primary_type) or base_palette.get(primary_type, CONFIG["DEFAULT_NODE_COLOR"])
        shape = CONFIG["NODE_TYPE_SHAPES"].get(primary_type, "dot")

    node_title = f"{label}\nTypes: {', '.join(canon_types)}"
    description = ""
    if "description" in metadata:
        if isinstance(metadata["description"], dict):
            description = metadata["description"].get("en", "")
        elif isinstance(metadata["description"], str):
            description = metadata["description"]
    if description:
        node_title += f"\nDescription: {description}"
    annotation_text = ""
    annotation_html = metadata.get("annotation_html")
    if isinstance(annotation_html, str) and annotation_html.strip():
        annotation_text = _clean_tooltip_text(_annotation_html_to_text(annotation_html))
    if not annotation_text and "annotation" in metadata and metadata["annotation"]:
        annotation_text = _clean_tooltip_text(str(metadata["annotation"]))
    if annotation_text:
        node_title += f"\nAnnotation: {annotation_text}"
    used_note_annotation = False
    if not annotation_text:
        note_annotation = _extract_annotation_from_notes(metadata)
        if note_annotation:
            annotation_text = _clean_tooltip_text(note_annotation)
            used_note_annotation = True
    notes = _extract_metadata_notes(metadata, skip_annotation=used_note_annotation)
    if notes:
        for idx, note in enumerate(notes[:2], start=1):
            trimmed = _clean_tooltip_text(note)
            if len(trimmed) > 240:
                trimmed = trimmed[:237].rstrip() + "..."
            label_prefix = "Note" if idx == 1 else f"Note {idx}"
            node_title += f"\n{label_prefix}: {trimmed}"

    size = custom_size if custom_size is not None else (22 if (search_nodes and node_id in search_nodes) else 16)
    node_color = _make_node_color(color)
    node_border_width = 3 if (search_nodes and node_id in search_nodes) else 2
    label_color = _pick_label_color(color)
    label_stroke = (
        "rgba(15, 23, 42, 0.65)"
        if label_color.lower() == "#f8f6f1"
        else "rgba(248, 244, 237, 0.92)"
    )
    visible_label = label if (show_labels and not smart_labels) else ""

    node_payload = {
        "label": visible_label,
        "fullLabel": label,
        "title": node_title,
        "color": node_color,
        "shape": shape,
        "size": size,
        "font": {
            "size": 20 if size >= 22 else 17,
            "face": VIS_FONT_FACE,
            "color": label_color,
            "strokeWidth": 3,
            "strokeColor": label_stroke,
            "align": "center",
            "vadjust": 0,
            "mod": "bold",
        },
        "borderWidth": node_border_width,
        "shadow": {"enabled": True, "color": "rgba(31, 42, 55, 0.18)", "size": 12, "x": 0, "y": 2},
        "widthConstraint": {"maximum": 190},
        "types": canon_types,
    }

    if type_icons:
        icon_map = CONFIG.get("NODE_TYPE_ICONS", {})
        icon_text = icon_map.get(primary_type, primary_type[:1] if primary_type else "?")
        node_payload["shape"] = "circularImage"
        node_payload["image"] = _type_icon_data(primary_type, color, icon_text)
        node_payload["size"] = max(size, 20)
        node_payload["iconText"] = icon_text
        node_payload["iconKey"] = primary_type

    net.add_node(node_id, **node_payload)

    logging.debug("Added node: %s (%s) with color %s and shape %s", label, node_id, color, shape)


def add_edge(
    net: Network,
    src: str,
    dst: str,
    relationship: str,
    id_to_label: Dict[str, str],
    search_nodes: Optional[List[str]] = None,
    custom_width: Optional[int] = None,
    custom_color: Optional[str] = None,
    relationship_colors: Optional[Dict[str, str]] = None,
    inferred: bool = False,
    weight: Optional[float] = None,
    edge_semantics: bool = True,
) -> None:
    is_search_edge = search_nodes is not None and (src in search_nodes or dst in search_nodes)
    palette = relationship_colors or {}
    base_palette = CONFIG["RELATIONSHIP_CONFIG"]
    base_color = (
        custom_color
        if custom_color is not None
        else palette.get(relationship) or base_palette.get(relationship, "#A9A9A9")
    )
    if edge_semantics and inferred and custom_color is None:
        base_color = _blend_hex(base_color, "#FFFFFF", 0.35)
    edge_color = _make_edge_color(base_color)
    label_text = re.sub(r"(?<!^)(?=[A-Z])", " ", relationship.replace("_", " ")).title()
    base_width = custom_width if custom_width is not None else (3.2 if is_search_edge else 2.2)
    weight_width = _edge_width_from_weight(weight, base_width) if edge_semantics else None
    width = max(base_width, weight_width) if weight_width is not None else base_width
    edge_title = f"{label_text}: {id_to_label.get(src, src)} -> {id_to_label.get(dst, dst)}"
    if edge_semantics and inferred:
        edge_title += " (inferred)"
    if edge_semantics and weight is not None:
        try:
            edge_title += f" | weight: {float(weight):.2f}"
        except (TypeError, ValueError):
            pass

    net.add_edge(
        src,
        dst,
        label=label_text,
        rel_key=relationship,
        color=edge_color,
        width=width,
        arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
        title=edge_title,
        dashes=[6, 5] if (edge_semantics and inferred) else False,
        font={
            "size": 11 if is_search_edge else 9,
            "align": "middle",
            "face": VIS_FONT_FACE,
            "color": "#2D3748",
            "strokeWidth": 3,
            "strokeColor": "rgba(248, 244, 237, 0.9)",
        },
        smooth={"enabled": True, "type": "dynamic"},
    )

    logging.debug("Added edge: %s --%s--> %s", src, label_text, dst)


def graph_css_block(reduce_motion: bool = False, graph_background: Optional[Dict[str, str]] = None) -> str:
    reduced_motion_css = ""
    if reduce_motion:
        reduced_motion_css = """
      * {
          animation: none !important;
          transition: none !important;
          scroll-behavior: auto !important;
      }
        """
    gradient, background_color = _build_graph_background(graph_background)
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      html, body {{
          margin: 0;
          padding: 0;
          font-family: '{APP_FONTS["body"]}', sans-serif;
          background: transparent;
          height: 100%;
          width: 100%;
      }}
      #mynetwork {{
          background: {gradient} !important;
          background-color: {background_color} !important;
          border: 1px solid rgba(33, 52, 71, 0.08) !important;
          border-radius: 16px !important;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7) !important;
          padding: 0 !important;
          box-sizing: border-box !important;
          margin: 0 !important;
          height: 100% !important;
          width: 100% !important;
      }}
      .card {{
          background: rgba(255, 255, 255, 0.96);
          border: 1px solid rgba(33, 52, 71, 0.1);
          border-radius: 18px;
          padding: 0;
          margin: 0;
          box-shadow: 0 16px 28px rgba(31, 42, 55, 0.1);
          box-sizing: border-box;
          height: {GRAPH_CARD_HEIGHT}px;
          display: flex;
          flex-direction: column;
      }}
      .graph-frame {{
          padding: {GRAPH_PANEL_PADDING}px;
          height: 100%;
          width: 100%;
          box-sizing: border-box;
          flex: 1 1 auto;
          min-height: 0;
      }}
      .card-body {{
          padding: 0;
          flex: 1 1 auto;
          height: 100%;
          min-height: 0;
      }}
      .vis-network {{
          border-radius: 16px;
      }}
      #mynetwork .vis-navigation {{
          bottom: {GRAPH_PANEL_PADDING}px !important;
          left: {GRAPH_PANEL_PADDING}px !important;
          right: {GRAPH_PANEL_PADDING}px !important;
      }}
      .vis-tooltip {{
          white-space: normal !important;
          max-width: 400px;
          min-width: 200px;
          font-family: '{APP_FONTS["body"]}', sans-serif;
          color: #1F2A37;
          background: rgba(255, 255, 255, 0.98);
          border: 1px solid rgba(61, 90, 128, 0.2);
          border-radius: 12px;
          padding: 10px 14px;
          box-shadow: 0 12px 24px rgba(31, 42, 55, 0.15);
          z-index: 10000;
      }}
      .vis-button {{
          background-color: rgba(255, 255, 255, 0.86);
          border: 1px solid rgba(61, 90, 128, 0.18);
          border-radius: 999px;
          box-shadow: 0 10px 20px rgba(31, 42, 55, 0.12);
          backdrop-filter: blur(6px);
          background-repeat: no-repeat;
          background-position: center !important;
          background-size: 18px 18px !important;
          transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
      }}
      .vis-button:hover {{
          background-color: rgba(255, 255, 255, 0.98);
          border-color: rgba(61, 90, 128, 0.32);
          box-shadow: 0 12px 22px rgba(31, 42, 55, 0.16);
          transform: translateY(-1px);
      }}
      .vis-button:active {{
          transform: translateY(0);
          box-shadow: 0 6px 14px rgba(31, 42, 55, 0.12);
      }}
      .vis-button.vis-up {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%2712%27%20y1%3D%2719%27%20x2%3D%2712%27%20y2%3D%275%27/%3E%3Cpolyline%20points%3D%275%2012%2012%205%2019%2012%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-down {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%2712%27%20y1%3D%275%27%20x2%3D%2712%27%20y2%3D%2719%27/%3E%3Cpolyline%20points%3D%275%2012%2012%2019%2019%2012%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-left {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%2719%27%20y1%3D%2712%27%20x2%3D%275%27%20y2%3D%2712%27/%3E%3Cpolyline%20points%3D%2712%205%205%2012%2012%2019%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-right {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%275%27%20y1%3D%2712%27%20x2%3D%2719%27%20y2%3D%2712%27/%3E%3Cpolyline%20points%3D%2712%205%2019%2012%2012%2019%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-zoomIn {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Ccircle%20cx%3D%2711%27%20cy%3D%2711%27%20r%3D%277%27/%3E%3Cline%20x1%3D%2721%27%20y1%3D%2721%27%20x2%3D%2716.65%27%20y2%3D%2716.65%27/%3E%3Cline%20x1%3D%2711%27%20y1%3D%278%27%20x2%3D%2711%27%20y2%3D%2714%27/%3E%3Cline%20x1%3D%278%27%20y1%3D%2711%27%20x2%3D%2714%27%20y2%3D%2711%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-zoomOut {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Ccircle%20cx%3D%2711%27%20cy%3D%2711%27%20r%3D%277%27/%3E%3Cline%20x1%3D%2721%27%20y1%3D%2721%27%20x2%3D%2716.65%27%20y2%3D%2716.65%27/%3E%3Cline%20x1%3D%278%27%20y1%3D%2711%27%20x2%3D%2714%27%20y2%3D%2711%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      .vis-button.vis-zoomExtends {{
          background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cpolyline%20points%3D%2715%203%2021%203%2021%209%27/%3E%3Cpolyline%20points%3D%279%2021%203%2021%203%2015%27/%3E%3Cline%20x1%3D%2721%27%20y1%3D%273%27%20x2%3D%2714%27%20y2%3D%2710%27/%3E%3Cline%20x1%3D%273%27%20y1%3D%2721%27%20x2%3D%2710%27%20y2%3D%2714%27/%3E%3C/svg%3E") !important;
          background-position: center !important;
      }}
      @media (prefers-reduced-motion: reduce) {{
          * {{
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
              scroll-behavior: auto !important;
          }}
      }}
      {reduced_motion_css}
    </style>
    """


def build_graph_custom_js(
    reduce_motion: bool,
    node_animations: Optional[List[str]],
    node_animation_strength: int,
    search_nodes: Optional[List[str]],
    path_nodes: Optional[List[str]] = None,
    centrality_values: Optional[Dict[str, float]] = None,
    graph_background: Optional[Dict[str, str]] = None,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    smart_labels: bool = False,
    show_labels: bool = True,
    label_zoom_threshold: float = 1.1,
    label_density_threshold: int = 60,
    focus_context: bool = False,
    node_count: int = 0,
) -> str:
    node_animation_strength = max(0, min(100, int(node_animation_strength)))
    if isinstance(node_animations, list):
        animation_values = [str(mode).lower() for mode in node_animations if mode]
    elif isinstance(node_animations, str):
        animation_values = [node_animations.lower()] if node_animations else []
    else:
        animation_values = []
    node_animations_js = json.dumps(animation_values)
    search_nodes_js = json.dumps(search_nodes or [])
    path_nodes_js = json.dumps(path_nodes or [])
    centrality_js = json.dumps(centrality_values or {})
    metadata_js = json.dumps(metadata_map or {})
    smart_labels_js = json.dumps(bool(smart_labels))
    show_labels_js = json.dumps(bool(show_labels))
    label_zoom_js = json.dumps(max(0.2, min(3.0, float(label_zoom_threshold))))
    label_density_js = json.dumps(max(10, int(label_density_threshold)))
    focus_context_js = json.dumps(bool(focus_context))
    node_count_js = json.dumps(int(node_count))
    gradient, background_color = _build_graph_background(graph_background)
    gradient_js = json.dumps(gradient)
    background_color_js = json.dumps(background_color)
    return f"""
    <script type="text/javascript">
      setTimeout(function() {{
          var container = document.getElementById('mynetwork');
          if (container) {{
              container.style.background = {gradient_js};
              container.style.backgroundColor = {background_color_js};
              container.addEventListener('contextmenu', function(e) {{
                  if (!network) {{
                      return;
                  }}
                  e.preventDefault();
                  var pointer = network.getPointer(e);
                  var nodeId = network.getNodeAt(pointer);
                  if (nodeId) {{
                      window.parent.postMessage({{type: 'OPEN_MODAL', node: nodeId}}, "*");
                  }}
              }});
          }}
          if (typeof network !== 'undefined') {{
              network.on("dragEnd", function(params) {{
                  var positions = {{}};
                  network.body.data.nodes.forEach(function(node) {{
                      positions[node.id] = {{x: node.x, y: node.y}};
                  }});
                  window.parent.postMessage({{type: 'UPDATE_POSITIONS', positions: positions}}, "*");
              }});
              var nodesData = network.body.data.nodes;
              var edgesData = network.body.data.edges;
              var metadataMap = {metadata_js};
              var nodeAnimations = {node_animations_js};
              var pulseStrength = {node_animation_strength};
              var reduceMotion = {str(reduce_motion).lower()};
              var searchNodes = {search_nodes_js};
              var pathNodes = {path_nodes_js};
              var centralityMap = {centrality_js};
              var smartLabels = {smart_labels_js};
              var showLabels = {show_labels_js};
              var labelZoomThreshold = {label_zoom_js};
              var labelDenseThreshold = {label_density_js};
              var focusContext = {focus_context_js};
              var nodeCount = {node_count_js};
              var detailPanel = document.getElementById("node-detail-panel");
              if (detailPanel && metadataMap && Object.keys(metadataMap).length) {{
                  var titleEl = document.getElementById("node-detail-title");
                  var idEl = document.getElementById("node-detail-id");
                  var typesEl = document.getElementById("node-detail-types");
                  var annotationWrap = document.getElementById("node-detail-annotation-wrap");
                  var annotationEl = document.getElementById("node-detail-annotation");
                  var propsWrap = document.getElementById("node-detail-props-wrap");
                  var propsEl = document.getElementById("node-detail-props");
                  var emptyEl = document.getElementById("node-detail-empty");

                  function setEmptyPanel() {{
                      detailPanel.classList.add("is-empty");
                      if (titleEl) titleEl.textContent = "Node details";
                      if (idEl) idEl.textContent = "";
                      if (typesEl) typesEl.textContent = "";
                      if (annotationWrap) annotationWrap.style.display = "none";
                      if (propsWrap) propsWrap.style.display = "none";
                      if (emptyEl) emptyEl.style.display = "block";
                  }}

                  function renderProperties(items, metadataHtml) {{
                      if (!propsEl) {{
                          return;
                      }}
                      if (metadataHtml) {{
                          propsEl.innerHTML = metadataHtml;
                          if (propsWrap) propsWrap.style.display = "block";
                          return;
                      }}
                      if (!items || !items.length) {{
                          propsEl.innerHTML = "";
                          if (propsWrap) propsWrap.style.display = "none";
                          return;
                      }}
                      var rows = items.map(function(item) {{
                          var key = item.key || "";
                          var value = item.value || "";
                          return "<div class='node-detail-row'>" +
                            "<span class='node-detail-key'>" + key + "</span>" +
                            "<span class='node-detail-value'>" + value + "</span>" +
                          "</div>";
                      }}).join("");
                      propsEl.innerHTML = rows;
                      if (propsWrap) propsWrap.style.display = "block";
                  }}

                  function renderAnnotation(htmlValue, textValue) {{
                      if (!annotationEl) {{
                          return;
                      }}
                      var html = htmlValue || "";
                      var text = textValue || "";
                      if (html) {{
                          annotationEl.innerHTML = html;
                          if (annotationWrap) annotationWrap.style.display = "block";
                          return;
                      }}
                      if (text) {{
                          annotationEl.innerHTML = "<p>" + text + "</p>";
                          if (annotationWrap) annotationWrap.style.display = "block";
                          return;
                      }}
                      annotationEl.innerHTML = "";
                      if (annotationWrap) annotationWrap.style.display = "none";
                  }}

                  function renderNodeDetails(nodeId) {{
                      if (!nodeId || !metadataMap[nodeId]) {{
                          setEmptyPanel();
                          return;
                      }}
                      var data = metadataMap[nodeId];
                      detailPanel.classList.remove("is-empty");
                      if (titleEl) titleEl.textContent = data.label || data.id || "Node";
                      if (idEl) idEl.textContent = data.id ? "ID: " + data.id : "";
                      var types = data.types || [];
                      if (typesEl) {{
                          typesEl.textContent = types.length ? types.join(" • ") : "";
                      }}
                      if (emptyEl) emptyEl.style.display = "none";
                      renderAnnotation(data.annotation_html, data.annotation_text);
                      renderProperties(data.properties || [], data.metadata_html || "");
                  }}

                  setEmptyPanel();
                  network.on("click", function(params) {{
                      var nodes = params.nodes || [];
                      if (nodes.length) {{
                          renderNodeDetails(nodes[0]);
                      }} else {{
                          setEmptyPanel();
                      }}
                  }});
                  network.on("selectNode", function(params) {{
                      var nodes = params.nodes || [];
                      if (nodes.length) {{
                          renderNodeDetails(nodes[0]);
                      }}
                  }});
                  network.on("deselectNode", function() {{
                      setEmptyPanel();
                  }});
              }}
              if (nodesData && smartLabels && showLabels) {{
                  var fullLabels = {{}};
                  var labelExpanded = null;
                  var selectedLabelSet = new Set();

                  nodesData.forEach(function(node) {{
                      fullLabels[node.id] = node.fullLabel || node.label || "";
                  }});

                  function refreshSelectedLabels() {{
                      if (!network.getSelectedNodes) {{
                          selectedLabelSet = new Set();
                          return;
                      }}
                      selectedLabelSet = new Set(network.getSelectedNodes() || []);
                  }}

                  function shouldExpandLabels() {{
                      if (!showLabels) {{
                          return false;
                      }}
                      if (!smartLabels) {{
                          return true;
                      }}
                      if (nodeCount > 0 && nodeCount <= labelDenseThreshold) {{
                          return true;
                      }}
                      if (!network.getScale) {{
                          return false;
                      }}
                      return network.getScale() >= labelZoomThreshold;
                  }}

                  function applyLabelState(force) {{
                      var expand = shouldExpandLabels();
                      if (!force && labelExpanded === expand) {{
                          return;
                      }}
                      labelExpanded = expand;
                      var updates = [];
                      nodesData.forEach(function(node) {{
                          var nextLabel = "";
                          if (expand) {{
                              nextLabel = fullLabels[node.id] || "";
                          }} else if (selectedLabelSet.has(node.id)) {{
                              nextLabel = fullLabels[node.id] || "";
                          }}
                          updates.push({{id: node.id, label: nextLabel}});
                      }});
                      if (updates.length) {{
                          nodesData.update(updates);
                      }}
                  }}

                  refreshSelectedLabels();
                  applyLabelState(true);

                  network.on("zoom", function() {{
                      applyLabelState(false);
                  }});
                  network.on("select", function() {{
                      refreshSelectedLabels();
                      applyLabelState(false);
                  }});
                  network.on("deselectNode", function() {{
                      refreshSelectedLabels();
                      applyLabelState(false);
                  }});
                  network.on("deselectEdge", function() {{
                      refreshSelectedLabels();
                      applyLabelState(false);
                  }});
                  network.on("hoverNode", function(params) {{
                      if (labelExpanded || !params || !params.node) {{
                          return;
                      }}
                      var nodeId = params.node;
                      var labelText = fullLabels[nodeId];
                      if (labelText) {{
                          nodesData.update([{{id: nodeId, label: labelText}}]);
                      }}
                  }});
                  network.on("blurNode", function(params) {{
                      if (labelExpanded || !params || !params.node) {{
                          return;
                      }}
                      var nodeId = params.node;
                      if (selectedLabelSet.has(nodeId)) {{
                          return;
                      }}
                      nodesData.update([{{id: nodeId, label: ""}}]);
                  }});
              }}
              if (nodesData && ((focusContext) || (!reduceMotion && pulseStrength > 0))) {{
                  var rawModes = Array.isArray(nodeAnimations) ? nodeAnimations : [nodeAnimations];
                  var modeSet = new Set(rawModes.map(function(mode) {{
                      return (mode || "").toString().toLowerCase();
                  }}).filter(function(mode) {{ return mode; }}));
                  if (modeSet.has("none") && modeSet.size > 1) {{
                      modeSet.delete("none");
                  }}
                  var focusContextActive = focusContext === true;
                  if ((modeSet.has("none") || modeSet.size === 0) && !focusContextActive) {{
                      return;
                  }}
                  var baseSizes = {{}};
                  var baseBorders = {{}};
                  var baseColors = {{}};
                  var baseFonts = {{}};
                  nodesData.forEach(function(node) {{
                      baseSizes[node.id] = node.size || 16;
                      baseBorders[node.id] = node.borderWidth || 2;
                      baseColors[node.id] = node.color || null;
                      if (node.font) {{
                          baseFonts[node.id] = Object.assign({{}}, node.font);
                      }}
                  }});
                  var baseEdgeWidths = {{}};
                  var baseEdgeColors = {{}};
                  var baseEdgeDashes = {{}};
                  var baseEdgeDashOffsets = {{}};
                  if (edgesData) {{
                      edgesData.forEach(function(edge) {{
                          baseEdgeWidths[edge.id] = edge.width || 2;
                          baseEdgeColors[edge.id] = edge.color || null;
                          if (edge.dashes !== undefined) {{
                              baseEdgeDashes[edge.id] = edge.dashes;
                          }}
                          if (edge.dashOffset !== undefined) {{
                              baseEdgeDashOffsets[edge.id] = edge.dashOffset;
                          }}
                      }});
                  }}
                  var allNodeIds = nodesData.getIds();
                  var allEdgeIds = edgesData ? edgesData.getIds() : [];
                  var activeSelection = network.getSelectedNodes ? network.getSelectedNodes() : [];
                  var activeEdges = [];
                  var neighborNodes = [];
                  var neighborEdges = [];
                  var strength = Math.max(0, Math.min(100, pulseStrength));
                  var amplitude = 0.08 + (strength / 100) * 0.22;
                  var speed = 0.06 + (strength / 100) * 0.14;
                  var interval = allNodeIds.length > 250 ? 160 : 110;
                  var edgeInterval = allEdgeIds.length > 250 ? 180 : 120;
                  if (!searchNodes || !searchNodes.length) {{
                      modeSet.delete("search");
                      modeSet.delete("search_ping");
                  }}
                  if (!pathNodes || pathNodes.length < 2) {{
                      modeSet.delete("path");
                  }}
                  if (!edgesData || !allEdgeIds.length) {{
                      modeSet.delete("flow");
                  }}
                  var hasSearch = modeSet.has("search");
                  var hasSearchPing = modeSet.has("search_ping");
                  var hasSelected = modeSet.has("selected");
                  var hasFocusMode = modeSet.has("focus");
                  var hasFocus = hasFocusMode || focusContextActive;
                  var hasNeighbors = modeSet.has("neighbors");
                  var hasPath = modeSet.has("path");
                  var hasCentrality = modeSet.has("centrality");
                  var hasAll = modeSet.has("all");
                  var hasFlow = modeSet.has("flow");
                  var hasPulse = hasSearch || hasSelected || hasFocusMode || hasNeighbors || hasPath || hasCentrality || hasAll;
                  if (!hasPulse && !hasFlow && !hasSearchPing && !focusContextActive) {{
                      return;
                  }}
                  function parseRgb(color) {{
                      if (!color || typeof color !== "string") {{
                          return null;
                      }}
                      if (color[0] === "#") {{
                          var hex = color.substring(1);
                          if (hex.length === 3) {{
                              hex = hex.split("").map(function(c) {{ return c + c; }}).join("");
                          }}
                          if (hex.length !== 6) {{
                              return null;
                          }}
                          var num = parseInt(hex, 16);
                          if (isNaN(num)) {{
                              return null;
                          }}
                          return {{
                              r: (num >> 16) & 255,
                              g: (num >> 8) & 255,
                              b: num & 255
                          }};
                      }}
                      var match = color.match(/rgba?\\(([^)]+)\\)/i);
                      if (!match) {{
                          return null;
                      }}
                      var parts = match[1].split(",").map(function(p) {{
                          return parseFloat(p.trim());
                      }});
                      if (parts.length < 3 || parts.some(function(p) {{ return isNaN(p); }})) {{
                          return null;
                      }}
                      return {{
                          r: Math.round(parts[0]),
                          g: Math.round(parts[1]),
                          b: Math.round(parts[2])
                      }};
                  }}
                  function colorToRgba(color, alpha) {{
                      var rgb = parseRgb(color);
                      if (!rgb) {{
                          return color;
                      }}
                      return "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", " + alpha + ")";
                  }}
                  function baseFontColor(nodeId) {{
                      if (baseFonts[nodeId] && baseFonts[nodeId].color) {{
                          return baseFonts[nodeId].color;
                      }}
                      return null;
                  }}
                  function withFontColor(nodeId, color) {{
                      if (!baseFonts[nodeId]) {{
                          return {{color: color}};
                      }}
                      var copy = {{}};
                      Object.keys(baseFonts[nodeId]).forEach(function(key) {{
                          copy[key] = baseFonts[nodeId][key];
                      }});
                      copy.color = color;
                      return copy;
                  }}
                  function makeDimNodeColor(nodeId, alpha) {{
                      var base = baseColors[nodeId];
                      if (!base) {{
                          return null;
                      }}
                      if (typeof base === "string") {{
                          return colorToRgba(base, alpha);
                      }}
                      var bg = base.background || "#CBD5E1";
                      var border = base.border || bg;
                      var dimBg = colorToRgba(bg, alpha);
                      var dimBorder = colorToRgba(border, Math.min(1, alpha + 0.12));
                      return {{
                          background: dimBg,
                          border: dimBorder,
                          highlight: {{
                              background: colorToRgba(bg, Math.min(1, alpha + 0.18)),
                              border: colorToRgba(border, Math.min(1, alpha + 0.24))
                          }},
                          hover: {{
                              background: colorToRgba(bg, Math.min(1, alpha + 0.12)),
                              border: colorToRgba(border, Math.min(1, alpha + 0.2))
                          }}
                      }};
                  }}
                  function makeDimEdgeColor(edgeId, alpha) {{
                      var base = baseEdgeColors[edgeId];
                      if (!base) {{
                          return {{
                              color: "rgba(45, 55, 72, " + alpha + ")",
                              opacity: alpha
                          }};
                      }}
                      if (typeof base === "string") {{
                          return {{
                              color: colorToRgba(base, alpha),
                              opacity: alpha
                          }};
                      }}
                      var baseColor = base.color || "#8C96A5";
                      return {{
                          color: colorToRgba(baseColor, alpha),
                          highlight: colorToRgba(base.highlight || baseColor, Math.min(1, alpha + 0.12)),
                          hover: colorToRgba(base.hover || baseColor, Math.min(1, alpha + 0.08)),
                          opacity: alpha
                      }};
                  }}
                  function resetNodes(ids, restoreColors) {{
                      if (!ids || !ids.length) {{
                          return;
                      }}
                      var updates = ids.map(function(id) {{
                          var update = {{
                              id: id,
                              size: baseSizes[id] || 16,
                              borderWidth: baseBorders[id] || 2
                          }};
                          if (restoreColors && baseColors[id]) {{
                              update.color = baseColors[id];
                          }}
                          if (restoreColors && baseFonts[id]) {{
                              var baseColor = baseFontColor(id);
                              if (baseColor) {{
                                  update.font = withFontColor(id, baseColor);
                              }}
                          }}
                          return update;
                      }});
                      nodesData.update(updates);
                  }}
                  function resetEdges(ids, restoreColors) {{
                      if (!edgesData || !ids || !ids.length) {{
                          return;
                      }}
                      var updates = ids.map(function(id) {{
                          var update = {{
                              id: id,
                              width: baseEdgeWidths[id] || 2
                          }};
                          if (restoreColors && baseEdgeColors[id]) {{
                              update.color = baseEdgeColors[id];
                          }}
                          if (restoreColors && baseEdgeDashes.hasOwnProperty(id)) {{
                              update.dashes = baseEdgeDashes[id];
                          }}
                          if (restoreColors && baseEdgeDashOffsets.hasOwnProperty(id)) {{
                              update.dashOffset = baseEdgeDashOffsets[id];
                          }}
                          return update;
                      }});
                      edgesData.update(updates);
                  }}
                  function updateActiveEdges() {{
                      if (!edgesData) {{
                          activeEdges = [];
                          return;
                      }}
                      var edgeSet = new Set();
                      if (network.getSelectedEdges) {{
                          (network.getSelectedEdges() || []).forEach(function(edgeId) {{
                              edgeSet.add(edgeId);
                          }});
                      }}
                      activeSelection.forEach(function(nodeId) {{
                          if (network.getConnectedEdges) {{
                              (network.getConnectedEdges(nodeId) || []).forEach(function(edgeId) {{
                                  edgeSet.add(edgeId);
                              }});
                          }}
                      }});
                      activeEdges = Array.from(edgeSet);
                  }}
                  function updateNeighborHalo() {{
                      if (!network.getConnectedNodes) {{
                          neighborNodes = [];
                          neighborEdges = [];
                          return;
                      }}
                      var neighborSet = new Set();
                      var edgeSet = new Set();
                      activeSelection.forEach(function(nodeId) {{
                          (network.getConnectedNodes(nodeId) || []).forEach(function(nei) {{
                              if (activeSelection.indexOf(nei) === -1) {{
                                  neighborSet.add(nei);
                              }}
                          }});
                          if (network.getConnectedEdges) {{
                              (network.getConnectedEdges(nodeId) || []).forEach(function(edgeId) {{
                                  edgeSet.add(edgeId);
                              }});
                          }}
                      }});
                      neighborNodes = Array.from(neighborSet);
                      neighborEdges = Array.from(edgeSet);
                  }}
                  function getSelectedNodeIds() {{
                      var selected = network.getSelectedNodes ? (network.getSelectedNodes() || []) : [];
                      if (network.getSelectedEdges && edgesData) {{
                          (network.getSelectedEdges() || []).forEach(function(edgeId) {{
                              var edge = edgesData.get(edgeId);
                              if (edge) {{
                                  selected.push(edge.from);
                                  selected.push(edge.to);
                              }}
                          }});
                      }}
                      return Array.from(new Set(selected));
                  }}
                  function applyFocusDimming() {{
                      var selectedSet = new Set(activeSelection || []);
                      var edgeSet = new Set(activeEdges || []);
                      var neighborSet = focusContextActive ? new Set(neighborNodes || []) : null;
                      var nodeUpdates = [];
                      allNodeIds.forEach(function(id) {{
                          var update = {{
                              id: id,
                              size: baseSizes[id] || 16,
                              borderWidth: baseBorders[id] || 2
                          }};
                          if (selectedSet.has(id)) {{
                              if (baseColors[id]) {{
                                  update.color = baseColors[id];
                              }}
                              var fontColor = baseFontColor(id);
                              if (fontColor) {{
                                  update.font = withFontColor(id, fontColor);
                              }}
                          }} else if (neighborSet && neighborSet.has(id)) {{
                              update.color = makeDimNodeColor(id, 0.4);
                              var midColor = baseFontColor(id);
                              if (midColor) {{
                                  update.font = withFontColor(id, colorToRgba(midColor, 0.45));
                              }}
                          }} else {{
                              update.color = makeDimNodeColor(id, 0.1);
                              var dimColor = baseFontColor(id);
                              if (dimColor) {{
                                  update.font = withFontColor(id, colorToRgba(dimColor, 0.2));
                              }}
                          }}
                          nodeUpdates.push(update);
                      }});
                      nodesData.update(nodeUpdates);
                      if (edgesData) {{
                          var edgeUpdates = [];
                          allEdgeIds.forEach(function(id) {{
                              var edgeUpdate = {{
                                  id: id,
                                  width: baseEdgeWidths[id] || 2
                              }};
                              if (edgeSet.has(id)) {{
                                  if (baseEdgeColors[id]) {{
                                      edgeUpdate.color = baseEdgeColors[id];
                                  }}
                              }} else {{
                                  edgeUpdate.color = makeDimEdgeColor(id, 0.08);
                                  edgeUpdate.width = Math.max(1, (baseEdgeWidths[id] || 2) * 0.7);
                              }}
                              edgeUpdates.push(edgeUpdate);
                          }});
                          edgesData.update(edgeUpdates);
                      }}
                  }}
                  function clearFocusDimming() {{
                      resetNodes(allNodeIds, true);
                      resetEdges(allEdgeIds, true);
                  }}
                  function syncSelection() {{
                      if (hasFocus) {{
                          activeSelection = getSelectedNodeIds();
                          updateActiveEdges();
                          updateNeighborHalo();
                          refreshSelectionSets();
                          if (activeSelection.length) {{
                              applyFocusDimming();
                          }} else {{
                              clearFocusDimming();
                          }}
                          return;
                      }}
                      resetNodes(activeSelection);
                      resetNodes(neighborNodes);
                      resetEdges(activeEdges);
                      resetEdges(neighborEdges);
                      activeSelection = getSelectedNodeIds();
                      updateActiveEdges();
                      updateNeighborHalo();
                      refreshSelectionSets();
                  }}
                  if (hasSelected || hasFocus || hasNeighbors) {{
                      syncSelection();
                      network.on("select", syncSelection);
                      network.on("deselectNode", syncSelection);
                      network.on("deselectEdge", syncSelection);
                      network.on("click", function(params) {{
                          if (!params.nodes || !params.nodes.length) {{
                              syncSelection();
                          }}
                      }});
                  }}
                  var pathEdgeIds = [];
                  if (hasPath && edgesData && pathNodes && pathNodes.length > 1) {{
                      var pathPairs = new Set();
                      for (var i = 0; i < pathNodes.length - 1; i += 1) {{
                          pathPairs.add(pathNodes[i] + "->" + pathNodes[i + 1]);
                      }}
                      edgesData.forEach(function(edge) {{
                          if (edge && pathPairs.has(edge.from + "->" + edge.to)) {{
                              pathEdgeIds.push(edge.id);
                          }}
                      }});
                  }}
                  var centralityValues = [];
                  if (centralityMap) {{
                      Object.keys(centralityMap).forEach(function(key) {{
                          var val = parseFloat(centralityMap[key]);
                          if (!isNaN(val)) {{
                              centralityValues.push(val);
                          }}
                      }});
                  }}
                  var minCentrality = centralityValues.length ? Math.min.apply(null, centralityValues) : 0;
                  var maxCentrality = centralityValues.length ? Math.max.apply(null, centralityValues) : 0;
                  var centralityRange = maxCentrality - minCentrality;
                  function centralityWeight(nodeId) {{
                      if (!centralityMap || centralityMap[nodeId] === undefined) {{
                          return 0.35;
                      }}
                      var raw = parseFloat(centralityMap[nodeId]);
                      if (isNaN(raw) || centralityRange <= 0) {{
                          return 0.35;
                      }}
                      return (raw - minCentrality) / centralityRange;
                  }}
                  var searchSet = new Set(searchNodes || []);
                  var pathSet = new Set(pathNodes || []);
                  var selectedSet = new Set(activeSelection || []);
                  var neighborSet = new Set(neighborNodes || []);
                  function refreshSelectionSets() {{
                      selectedSet = new Set(activeSelection || []);
                      neighborSet = new Set(neighborNodes || []);
                  }}
                  function nodeAmplitude(nodeId) {{
                      var amp = 0;
                      if (hasAll) {{
                          amp = Math.max(amp, amplitude);
                      }}
                      if (hasSearch && searchSet.has(nodeId)) {{
                          amp = Math.max(amp, amplitude);
                      }}
                      if ((hasSelected || hasFocus) && selectedSet.has(nodeId)) {{
                          amp = Math.max(amp, amplitude);
                      }}
                      if (hasNeighbors && neighborSet.has(nodeId)) {{
                          amp = Math.max(amp, amplitude);
                      }}
                      if (hasPath && pathSet.has(nodeId)) {{
                          amp = Math.max(amp, amplitude);
                      }}
                      if (hasCentrality) {{
                          var cAmp = amplitude * (0.35 + 0.85 * centralityWeight(nodeId));
                          amp = Math.max(amp, cAmp);
                      }}
                      return amp;
                  }}
                  function getTargets() {{
                      if (hasAll || hasCentrality) {{
                          return allNodeIds;
                      }}
                      var targetSet = new Set();
                      if (hasSearch) {{
                          (searchNodes || []).forEach(function(id) {{ targetSet.add(id); }});
                      }}
                      if (hasSelected || hasFocus) {{
                          (activeSelection || []).forEach(function(id) {{ targetSet.add(id); }});
                      }}
                      if (hasNeighbors) {{
                          (neighborNodes || []).forEach(function(id) {{ targetSet.add(id); }});
                      }}
                      if (hasPath) {{
                          (pathNodes || []).forEach(function(id) {{ targetSet.add(id); }});
                      }}
                      return Array.from(targetSet);
                  }}
                  function getEdgeTargets() {{
                      var edgeSet = new Set();
                      if (hasSelected || hasFocus) {{
                          (activeEdges || []).forEach(function(id) {{ edgeSet.add(id); }});
                      }}
                      if (hasNeighbors) {{
                          (neighborEdges || []).forEach(function(id) {{ edgeSet.add(id); }});
                      }}
                      if (hasPath) {{
                          (pathEdgeIds || []).forEach(function(id) {{ edgeSet.add(id); }});
                      }}
                      return Array.from(edgeSet);
                  }}
                  function startEdgeFlow() {{
                      if (!edgesData || !allEdgeIds.length) {{
                          return;
                      }}
                      var dashLength = 6 + (strength / 100) * 8;
                      var gapLength = 10 + (strength / 100) * 10;
                      var offset = 0;
                      var flowSpeed = 0.6 + (strength / 100) * 1.4;
                      edgesData.update(allEdgeIds.map(function(id) {{
                          return {{
                              id: id,
                              dashes: [dashLength, gapLength]
                          }};
                      }}));
                      setInterval(function() {{
                          offset = (offset + flowSpeed) % (dashLength + gapLength);
                          var updates = allEdgeIds.map(function(id) {{
                              return {{
                                  id: id,
                                  dashes: [dashLength, gapLength],
                                  dashOffset: -offset
                              }};
                          }});
                          edgesData.update(updates);
                      }}, edgeInterval);
                  }}
                  function runSearchPing() {{
                      var targets = searchNodes || [];
                      if (!targets || !targets.length) {{
                          return;
                      }}
                      var totalTicks = 12;
                      var tick = 0;
                      var stagger = targets.length > 1 ? 0.35 : 0;
                      var timer = setInterval(function() {{
                          var progress = tick / (totalTicks - 1);
                          var updates = [];
                          targets.forEach(function(id, idx) {{
                              var delay = stagger
                                  ? (idx / Math.max(1, targets.length - 1)) * stagger
                                  : 0;
                              var local = Math.max(0, Math.min(1, (progress - delay) / Math.max(0.01, 1 - stagger)));
                              var localWave = Math.sin(Math.PI * local);
                              if (baseSizes[id]) {{
                                  updates.push({{
                                      id: id,
                                      size: baseSizes[id] * (1 + localWave * amplitude * 1.4),
                                      borderWidth: (baseBorders[id] || 2) * (1 + localWave * amplitude * 2.0)
                                  }});
                              }}
                          }});
                          if (updates.length) {{
                              nodesData.update(updates);
                          }}
                          tick += 1;
                          if (tick >= totalTicks) {{
                              clearInterval(timer);
                              resetNodes(targets);
                          }}
                      }}, interval);
                  }}
                  function startPulse() {{
                      var phase = 0;
                      setInterval(function() {{
                          var targets = getTargets();
                          var edgeTargets = getEdgeTargets();
                          if ((!targets || targets.length === 0) && (!edgeTargets || edgeTargets.length === 0)) {{
                              return;
                          }}
                          phase += speed;
                          var wave = Math.sin(phase);
                          var edgeAmplitude = 0.25 + (strength / 100) * 0.55;
                          var edgeScale = 1 + wave * edgeAmplitude;
                          var updates = [];
                          targets.forEach(function(id) {{
                              if (baseSizes[id]) {{
                                  var nodeAmp = nodeAmplitude(id);
                                  if (nodeAmp > 0) {{
                                      var nodeScale = 1 + wave * nodeAmp;
                                      var borderScale = 1 + wave * (nodeAmp * 2.2);
                                      updates.push({{
                                          id: id,
                                          size: baseSizes[id] * nodeScale,
                                          borderWidth: (baseBorders[id] || 2) * borderScale
                                      }});
                                  }}
                              }}
                          }});
                          if (updates.length) {{
                              nodesData.update(updates);
                          }}
                          if (edgesData && edgeTargets && edgeTargets.length) {{
                              var edgeUpdates = [];
                              edgeTargets.forEach(function(id) {{
                                  var baseWidth = baseEdgeWidths[id] || 2;
                                  edgeUpdates.push({{
                                      id: id,
                                      width: Math.max(1, baseWidth * edgeScale)
                                  }});
                              }});
                              if (edgeUpdates.length) {{
                                  edgesData.update(edgeUpdates);
                              }}
                          }}
                      }}, interval);
                  }}
                  if (hasFlow) {{
                      startEdgeFlow();
                  }}
                  if (hasSearchPing) {{
                      runSearchPing();
                  }}
                  if (hasPulse) {{
                      startPulse();
                  }}
              }}
          }}
      }}, 1000);
    </script>
    """


# pylint: disable=too-many-locals,too-many-branches,too-many-statements

def build_graph(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    selected_relationships: List[str],
    search_nodes: Optional[List[str]] = None,
    node_positions: Optional[Dict[str, Dict[str, float]]] = None,
    show_labels: bool = True,
    smart_labels: bool = False,
    label_zoom_threshold: float = 1.1,
    filtered_nodes: Optional[Set[str]] = None,
    community_detection: bool = False,
    centrality: Optional[Dict[str, Dict[str, float]]] = None,
    path_nodes: Optional[List[str]] = None,
    graph_version: Optional[int] = None,
    reduce_motion: bool = False,
    motion_intensity: int = 50,
    node_animations: Optional[List[str]] = None,
    node_animation_strength: int = 30,
    node_type_colors: Optional[Dict[str, str]] = None,
    relationship_colors: Optional[Dict[str, str]] = None,
    graph_background: Optional[Dict[str, str]] = None,
    community_colors: Optional[List[str]] = None,
    community_locks: Optional[Dict[int, str]] = None,
    focus_context: bool = False,
    edge_semantics: bool = True,
    type_icons: bool = False,
) -> Network:
    net = Network(
        height=f"{GRAPH_CANVAS_HEIGHT}px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="transparent",
        font_color="#1F2A37",
    )

    added_nodes: Set[str] = set()
    edge_set: Set[Tuple[str, str, str]] = set()
    path_edge_set = set(zip(path_nodes, path_nodes[1:])) if path_nodes else set()

    for node in graph_data.nodes:
        if filtered_nodes is not None and node.id not in filtered_nodes:
            logging.debug("Skipping node %s due to filtering", node.id)
            continue

        custom_size = None
        if centrality and node.id in centrality:
            custom_size = int(15 + centrality[node.id]["degree"] * 30)
        if path_nodes and node.id in path_nodes:
            custom_size = max(custom_size or 15, 25)

        if node.id not in added_nodes:
            add_node(
                net,
                node.id,
                id_to_label.get(node.id, node.id),
                node.types,
                node.metadata,
                search_nodes,
                show_labels,
                smart_labels,
                custom_size,
                node_type_colors,
                type_icons,
            )
            added_nodes.add(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            if edge.relationship not in selected_relationships:
                continue
            if filtered_nodes is not None and (
                edge.source not in filtered_nodes or edge.target not in filtered_nodes
            ):
                logging.debug(
                    "Skipping edge %s --%s--> %s due to filtering",
                    edge.source,
                    edge.relationship,
                    edge.target,
                )
                continue
            if edge.target not in added_nodes:
                target_label = id_to_label.get(edge.target, _local_name(edge.target))
                add_node(
                    net,
                    edge.target,
                    target_label,
                    ["Unknown"],
                    {},
                    search_nodes,
                    show_labels,
                    smart_labels,
                    None,
                    node_type_colors,
                    type_icons,
                )
                added_nodes.add(edge.target)

            if (edge.source, edge.target, edge.relationship) not in edge_set:
                if path_nodes and (edge.source, edge.target) in path_edge_set:
                    custom_width = 4
                    custom_color = "#E07A5F"
                else:
                    custom_width = None
                    custom_color = None

                add_edge(
                    net,
                    edge.source,
                    edge.target,
                    edge.relationship,
                    id_to_label,
                    search_nodes,
                    custom_width,
                    custom_color,
                    relationship_colors,
                    inferred=getattr(edge, "inferred", False),
                    weight=getattr(edge, "weight", None),
                    edge_semantics=edge_semantics,
                )
                edge_set.add((edge.source, edge.target, edge.relationship))

    node_count = len(net.nodes)
    node_font_size = 14 if node_count <= 40 else 12 if node_count <= 90 else 10
    edge_font_size = 11 if node_count <= 40 else 9 if node_count <= 90 else 8
    label_density_threshold = 60

    intensity = max(0, min(100, int(motion_intensity)))
    if reduce_motion:
        intensity = min(intensity, 30)
    intensity_scale = intensity / 100.0
    speed_scale = 0.25 + 0.75 * intensity_scale
    spring_length_scale = 1.15 - 0.35 * intensity_scale
    stabilization_iterations = int(40 + 160 * intensity_scale)
    max_velocity = 4 + 46 * intensity_scale
    min_velocity = 0.02 + 0.18 * intensity_scale
    damping = 0.92 - 0.3 * intensity_scale
    timestep = 0.15 + 0.55 * intensity_scale
    physics_defaults = CONFIG["PHYSICS_DEFAULTS"]
    gravity_base = float(st.session_state.physics_params.get("gravity", physics_defaults["gravity"]))
    central_gravity_base = float(
        st.session_state.physics_params.get("centralGravity", physics_defaults["centralGravity"])
    )
    spring_length_base = float(
        st.session_state.physics_params.get("springLength", physics_defaults["springLength"])
    )
    spring_strength_base = float(
        st.session_state.physics_params.get("springStrength", physics_defaults["springStrength"])
    )
    gravity = gravity_base * speed_scale
    central_gravity = central_gravity_base * speed_scale
    spring_length = spring_length_base * spring_length_scale
    spring_strength = spring_strength_base * speed_scale

    default_options = {
        "layout": {"improvedLayout": True, "randomSeed": 23},
        "nodes": {
            "font": {
                "size": node_font_size,
                "face": VIS_FONT_FACE,
                "color": "#1F2A37",
                "strokeWidth": 3,
                "strokeColor": "rgba(248, 244, 237, 0.9)",
                "mod": "bold",
            },
            "shapeProperties": {"borderDashes": False, "useBorderWithImage": True},
            "shadow": {"enabled": True, "color": "rgba(31, 42, 55, 0.12)", "size": 10, "x": 0, "y": 2},
        },
        "edges": {
            "font": {
                "size": edge_font_size,
                "face": VIS_FONT_FACE,
                "align": "middle",
                "color": "#2D3748",
                "strokeWidth": 3,
                "strokeColor": "rgba(248, 244, 237, 0.9)",
            },
            "smooth": {"type": "dynamic", "roundness": 0.35},
            "width": 2,
            "selectionWidth": 2.4,
            "hoverWidth": 2.8,
        },
        "physics": {
            "enabled": bool(st.session_state.enable_physics),
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": gravity,
                "centralGravity": central_gravity,
                "springLength": spring_length,
                "springConstant": spring_strength,
                "damping": damping,
                "avoidOverlap": 0.8,
            },
            "minVelocity": min_velocity,
            "maxVelocity": max_velocity,
            "timestep": timestep,
            "stabilization": {"enabled": True, "iterations": stabilization_iterations, "updateInterval": 20, "fit": True},
        },
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "navigationButtons": True,
            "zoomView": True,
            "dragNodes": True,
            "multiselect": True,
            "selectConnectedEdges": True,
            "tooltipDelay": 120,
            "hideEdgesOnDrag": True,
        },
    }
    net.options = default_options

    if community_detection:
        G_comm = nx.Graph()
        for node in net.nodes:
            if "id" in node:
                G_comm.add_node(node["id"])
        for edge in net.edges:
            if "from" in edge and "to" in edge:
                G_comm.add_edge(edge["from"], edge["to"])
        community_map: Optional[Dict[str, int]] = None
        if louvain_installed:
            try:
                partition = community_louvain.best_partition(G_comm, random_state=0)
            except TypeError:
                partition = community_louvain.best_partition(G_comm)
            community_map = partition
            logging.info("Community partition: %s", partition)
        else:
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(G_comm)
                community_map = {}
                for idx, group in enumerate(communities):
                    for node_id in group:
                        community_map[node_id] = idx
                st.info("Louvain not available; using modularity communities instead.")
            except Exception:
                st.info("Community detection not available.")

        if community_map:
            community_groups: Dict[int, List[str]] = {}
            for node_id, cid in community_map.items():
                community_groups.setdefault(cid, []).append(str(node_id))
            ordered_ids = sorted(
                community_groups.keys(),
                key=lambda cid: (
                    -len(community_groups[cid]),
                    min(community_groups[cid]) if community_groups[cid] else "",
                    str(cid),
                ),
            )
            community_count = len(ordered_ids)
            locked_assignments: Dict[int, int] = {}
            used_slots: Set[int] = set()
            if isinstance(community_locks, dict):
                for slot_index, node_id in community_locks.items():
                    try:
                        slot_idx = int(slot_index)
                    except (TypeError, ValueError):
                        continue
                    if slot_idx < 0 or slot_idx >= community_count:
                        continue
                    if not node_id:
                        continue
                    cid = community_map.get(str(node_id))
                    if cid is None or cid in locked_assignments or slot_idx in used_slots:
                        continue
                    locked_assignments[cid] = slot_idx
                    used_slots.add(slot_idx)

            available_slots = [idx for idx in range(community_count) if idx not in used_slots]
            id_to_index: Dict[int, int] = dict(locked_assignments)
            slot_iter = iter(available_slots)
            for cid in ordered_ids:
                if cid in id_to_index:
                    continue
                id_to_index[cid] = next(slot_iter)

            default_palette = CONFIG.get("COMMUNITY_COLORS") or _COMMUNITY_DEFAULTS
            base_palette = _coerce_color_list(community_colors, list(default_palette))
            community_palette = _ensure_palette_size(base_palette, community_count)
            for node in net.nodes:
                    node_id = node.get("id")
                    if node_id in community_map:
                        idx = id_to_index[community_map[node_id]]
                        base_color = community_palette[idx]
                        node["color"] = _make_node_color(base_color)
                    if node.get("shape") == "circularImage" and node.get("iconText"):
                        node["image"] = _type_icon_data(
                            str(node.get("iconKey") or "Unknown"),
                            base_color,
                            str(node.get("iconText")),
                        )
                    node_font = node.get("font", {})
                    label_color = _pick_label_color(base_color)
                    node_font["color"] = label_color
                    if label_color.lower() == "#f8f6f1":
                        try:
                            existing_stroke = int(node_font.get("strokeWidth", 0) or 0)
                        except (TypeError, ValueError):
                            existing_stroke = 0
                        node_font["strokeWidth"] = max(existing_stroke, 3)
                        node_font["strokeColor"] = "rgba(15, 23, 42, 0.65)"
                    node["font"] = node_font
            net._community_palette = community_palette
            net._community_count = community_count
        net._community_applied = bool(community_map)
    else:
        net._community_applied = False

    centrality_values = None
    if centrality:
        centrality_values = {node_id: float(values.get("degree", 0.0)) for node_id, values in centrality.items()}
    custom_js = build_graph_custom_js(
        reduce_motion=reduce_motion,
        node_animations=node_animations,
        node_animation_strength=node_animation_strength,
        search_nodes=search_nodes,
        path_nodes=path_nodes,
        centrality_values=centrality_values,
        graph_background=graph_background,
        smart_labels=smart_labels,
        show_labels=show_labels,
        label_zoom_threshold=label_zoom_threshold,
        label_density_threshold=label_density_threshold,
        focus_context=focus_context,
        node_count=node_count,
    )
    custom_css = graph_css_block(reduce_motion=reduce_motion, graph_background=graph_background)
    if node_positions:
        for node in net.nodes:
            pos = node_positions.get(node.get("id"))
            if pos:
                node["x"] = pos["x"]
                node["y"] = pos["y"]
                node["fixed"] = True
                node["physics"] = False

    base_html = net.generate_html()
    base_html = base_html.replace(
        '<div id="mynetwork" class="card-body"></div>',
        '<div class="graph-frame"><div id="mynetwork" class="card-body"></div></div>',
        1,
    )
    base_html = base_html.replace(
        '<div id="mynetwork" class="card-body"></div>',
        '<div class="graph-frame"><div id="mynetwork" class="card-body"></div></div>',
        1,
    )
    html = base_html.replace("</head>", custom_css + "</head>", 1)
    html = html.replace("</body>", custom_js + "</body>", 1)
    net.html = html
    return net


def convert_graph_to_jsonld(net: Network) -> Dict[str, Any]:
    nodes_dict = {}
    for node in net.nodes:
        node_id = node.get("id")
        nodes_dict[node_id] = {
            "@id": node_id,
            "label": node.get("label", ""),
            "x": node.get("x"),
            "y": node.get("y"),
        }
        if "types" in node:
            nodes_dict[node_id]["type"] = node["types"]

    for edge in net.edges:
        source = edge.get("from")
        target = edge.get("to")
        rel_key = edge.get("rel_key") or edge.get("label", "").strip().replace(" ", "")
        if not rel_key:
            continue
        prop = "ex:" + rel_key
        triple = {"@id": target}
        if prop in nodes_dict[source]:
            if isinstance(nodes_dict[source][prop], list):
                nodes_dict[source][prop].append(triple)
            else:
                nodes_dict[source][prop] = [nodes_dict[source][prop], triple]
        else:
            nodes_dict[source][prop] = triple

    return {
        "@context": {
            "label": "http://www.w3.org/2000/01/rdf-schema#label",
            "x": "http://example.org/x",
            "y": "http://example.org/y",
            "type": "@type",
            "ex": "http://example.org/",
        },
        "@graph": list(nodes_dict.values()),
    }


def convert_graph_to_gexf(graph_data: GraphData, id_to_label: Dict[str, str]) -> str:
    gexf_graph = nx.DiGraph()
    for node in graph_data.nodes:
        node_id = str(node.id)
        label = id_to_label.get(node.id, node.label or node.id)
        types = [canonical_type(t) for t in (node.types or [])] or ["Unknown"]
        inferred_types = _extract_inferred_types(node.metadata)
        node_attrs = {
            "label": str(label),
            "types": ", ".join(types),
        }
        if inferred_types:
            node_attrs["inferredTypes"] = ", ".join(_shorten_iri(t) for t in inferred_types)
        gexf_graph.add_node(node_id, **node_attrs)

    for node in graph_data.nodes:
        for edge in node.edges:
            if not edge.source or not edge.target:
                continue
            edge_attrs = {"relationship": str(edge.relationship)}
            if getattr(edge, "inferred", False):
                edge_attrs["inferred"] = True
            if getattr(edge, "weight", None) is not None:
                try:
                    edge_attrs["weight"] = float(edge.weight)
                except (TypeError, ValueError):
                    pass
            gexf_graph.add_edge(str(edge.source), str(edge.target), **edge_attrs)

    output = io.BytesIO()
    nx.write_gexf(gexf_graph, output, encoding="utf-8")
    return output.getvalue().decode("utf-8")


def create_legends(
    rel_colors: Dict[str, str],
    node_colors: Dict[str, str],
    community_colors: Optional[List[str]] = None,
    community_labels: Optional[List[str]] = None,
) -> str:
    rel_items = "".join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>"
        f"<span>{rel.replace('_', ' ').title()}</span></div>"
        for rel, color in rel_colors.items()
    )
    node_items = "".join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>"
        f"<span>{ntype}</span></div>"
        for ntype, color in node_colors.items()
    )
    community_block = ""
    if community_colors:
        community_block = (
            "<div class='legend-card'>"
            "<div class='legend-title'>Communities</div>"
            f"<div class='legend-grid legend-grid-tight'>"
            f"{create_community_legend(community_colors, community_labels)}"
            "</div>"
            "</div>"
        )
    return (
        "<div class='legend-wrap'>"
        f"{community_block}"
        "<div class='legend-card'>"
        "<div class='legend-title'>Relationships</div>"
        f"<div class='legend-grid'>{rel_items}</div>"
        "</div>"
        "<div class='legend-card'>"
        "<div class='legend-title'>Node Types</div>"
        f"<div class='legend-grid legend-grid-tight'>{node_items}</div>"
        "</div>"
        "</div>"
    )


def create_relationship_legend(rel_colors: Dict[str, str]) -> str:
    rel_items = "".join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>"
        f"<span>{rel.replace('_', ' ').title()}</span></div>"
        for rel, color in rel_colors.items()
    )
    return (
        "<div class='legend-card'>"
        "<div class='legend-title'>Relationships</div>"
        f"<div class='legend-grid'>{rel_items}</div>"
        "</div>"
    )


def create_node_type_legend(node_colors: Dict[str, str]) -> str:
    node_items = "".join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>"
        f"<span>{ntype}</span></div>"
        for ntype, color in node_colors.items()
    )
    return (
        "<div class='legend-card'>"
        "<div class='legend-title'>Node Types</div>"
        f"<div class='legend-grid legend-grid-tight'>{node_items}</div>"
        "</div>"
    )


def create_community_legend(
    community_colors: List[str], community_labels: Optional[List[str]] = None
) -> str:
    if not community_colors:
        return ""
    labels = community_labels or []
    items = "".join(
        f"<div class='legend-item'><span class='legend-swatch' style='background:{color};'></span>"
        f"<span>{html_lib.escape(labels[idx].strip()) if idx < len(labels) and str(labels[idx]).strip() else f'Community {idx + 1}'}</span></div>"
        for idx, color in enumerate(community_colors)
    )
    return f"{items}"


def export_css_block() -> str:
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      :root {{
          --ink-1: #1E2A35;
          --ink-2: #4B5A6A;
          --ink-3: #6B7A8C;
          --panel: #FFFFFF;
          --panel-border: rgba(33, 52, 71, 0.12);
          --accent-2: #2D4F6A;
          --font-display: '{APP_FONTS["display"]}', serif;
      }}
      html, body {{
          margin: 0;
          padding: 0;
          background: linear-gradient(135deg, #F6F2EA 0%, #EEE7DE 100%);
          font-family: '{APP_FONTS["body"]}', sans-serif;
          color: var(--ink-1);
      }}
      body {{
          padding: 28px;
      }}
      .export-page {{
          max-width: 1200px;
          margin: 0 auto;
      }}
      .export-hero {{
          background: rgba(255, 255, 255, 0.95);
          border: 1px solid var(--panel-border);
          border-radius: 20px;
          padding: 22px 24px;
          box-shadow: 0 18px 32px rgba(31, 42, 55, 0.12);
      }}
      .export-title {{
          font-family: '{APP_FONTS["display"]}', serif;
          font-size: 32px;
          margin: 0;
      }}
      .export-subtitle {{
          margin-top: 6px;
          color: var(--ink-2);
          font-size: 0.95rem;
      }}
      .export-stats {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          gap: 10px;
          margin-top: 16px;
      }}
      .export-stat {{
          background: var(--panel);
          border: 1px solid rgba(33, 52, 71, 0.12);
          border-radius: 12px;
          padding: 10px 12px;
          display: flex;
          justify-content: space-between;
          align-items: baseline;
      }}
      .export-stat span {{
          color: var(--ink-3);
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 600;
      }}
      .export-stat strong {{
          font-size: 1.2rem;
          color: var(--ink-1);
      }}
      .export-notes {{
          margin-top: 12px;
          color: var(--ink-2);
          font-size: 0.9rem;
      }}
      .export-legend {{
          margin-top: 16px;
          margin-bottom: 16px;
      }}
      .export-section-title {{
          font-family: '{APP_FONTS["display"]}', serif;
          font-size: 1.15rem;
          margin: 0 0 0.6rem;
      }}
      .legend-wrap {{
          display: grid;
          gap: 1rem;
      }}
      .legend-card {{
          background: rgba(255, 255, 255, 0.92);
          border: 1px solid rgba(33, 52, 71, 0.12);
          border-radius: 14px;
          padding: 0.85rem 1rem;
          box-shadow: 0 12px 22px rgba(31, 42, 55, 0.08);
      }}
      .legend-title {{
          font-weight: 700;
          margin-bottom: 0.6rem;
      }}
      .legend-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          gap: 0.35rem 0.6rem;
      }}
      .legend-grid-tight {{
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      }}
      .legend-item {{
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: var(--ink-2);
          font-size: 0.9rem;
      }}
      .legend-swatch {{
          width: 10px;
          height: 10px;
          border-radius: 999px;
          border: 1px solid rgba(0, 0, 0, 0.1);
          display: inline-block;
      }}
      .card {{
          background: rgba(255, 255, 255, 0.96);
          border: 1px solid rgba(33, 52, 71, 0.1);
          border-radius: 18px;
          padding: 0;
          margin: 12px 0 18px;
          box-shadow: 0 16px 28px rgba(31, 42, 55, 0.1);
      }}
      .card-body {{
          padding: 0;
      }}
      .export-page #mynetwork {{
          margin: 0;
          border: 1px solid rgba(33, 52, 71, 0.08);
          border-radius: 16px;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
          padding: 0;
          box-sizing: border-box;
      }}
      .graph-frame {{
          position: relative;
      }}
      .node-detail-panel {{
          position: absolute;
          left: 50%;
          bottom: 18px;
          transform: translateX(-50%);
          width: min(680px, 92%);
          max-height: 46%;
          overflow: auto;
          background: rgba(255, 255, 255, 0.96);
          border: 1px solid rgba(33, 52, 71, 0.14);
          border-radius: 16px;
          padding: 14px 16px;
          box-shadow: 0 18px 32px rgba(31, 42, 55, 0.18);
          backdrop-filter: blur(6px);
      }}
      .node-detail-panel.is-empty {{
          opacity: 0.8;
      }}
      .node-detail-title {{
          font-family: '{APP_FONTS["display"]}', serif;
          font-size: 1.05rem;
          margin: 0 0 0.2rem 0;
      }}
      .node-detail-id {{
          color: var(--ink-3);
          font-size: 0.78rem;
          margin-bottom: 0.35rem;
      }}
      .node-detail-types {{
          color: var(--ink-2);
          font-size: 0.85rem;
          margin-bottom: 0.6rem;
      }}
      .node-detail-section {{
          margin-top: 0.65rem;
      }}
      .node-detail-section-title {{
          font-size: 0.7rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-3);
          margin-bottom: 0.35rem;
          font-weight: 600;
      }}
      .node-detail-annotation p {{
          margin: 0 0 0.4rem 0;
      }}
      .node-detail-annotation p:last-child {{
          margin-bottom: 0;
      }}
      .node-detail-row {{
          display: grid;
          grid-template-columns: minmax(110px, 0.35fr) 1fr;
          gap: 8px;
          padding: 0.35rem 0;
          border-bottom: 1px dashed rgba(33, 52, 71, 0.12);
      }}
      .node-detail-row:last-child {{
          border-bottom: none;
      }}
      .node-detail-key {{
          color: var(--ink-3);
          font-size: 0.78rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
      }}
      .node-detail-value {{
          color: var(--ink-1);
          font-size: 0.86rem;
          word-break: break-word;
      }}
      .node-detail-empty {{
          color: var(--ink-2);
          font-size: 0.85rem;
      }}
      .meta-panel {{
          background: rgba(255, 255, 255, 0.94);
          border: 1px solid rgba(33, 52, 71, 0.12);
          border-radius: 16px;
          padding: 1rem 1.1rem;
          box-shadow: 0 12px 22px rgba(31, 42, 55, 0.08);
          margin-top: 0.65rem;
      }}
      .meta-header {{
          font-family: var(--font-display);
          font-size: 1.05rem;
          color: var(--ink-1);
          margin-bottom: 0.75rem;
      }}
      .meta-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 0.75rem;
      }}
      .meta-card {{
          background: linear-gradient(145deg, rgba(255, 255, 255, 0.98), rgba(248, 244, 237, 0.92));
          border: 1px solid rgba(33, 52, 71, 0.12);
          border-radius: 14px;
          padding: 0.75rem 0.9rem;
          box-shadow: 0 10px 18px rgba(31, 42, 55, 0.08);
          min-height: 88px;
      }}
      .meta-card-more {{
          background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(247, 240, 232, 0.96));
      }}
      .meta-key {{
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-3);
          font-weight: 600;
          margin-bottom: 0.35rem;
      }}
      .meta-value {{
          font-size: 0.95rem;
          color: var(--ink-1);
          line-height: 1.35;
          word-break: break-word;
      }}
      .meta-muted {{
          color: var(--ink-3);
          font-size: 0.85rem;
      }}
      .meta-chips {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
      }}
      .meta-chip {{
          background: rgba(45, 79, 106, 0.1);
          color: var(--ink-1);
          border: 1px solid rgba(33, 52, 71, 0.16);
          padding: 0.2rem 0.45rem;
          border-radius: 999px;
          font-size: 0.78rem;
          display: inline-flex;
          align-items: center;
          text-decoration: none;
          line-height: 1.2;
      }}
      .meta-chip:hover {{
          border-color: rgba(33, 52, 71, 0.28);
      }}
      .meta-chip-primary {{
          background: rgba(208, 106, 76, 0.16);
          border-color: rgba(208, 106, 76, 0.32);
      }}
      .meta-more {{
          background: rgba(208, 106, 76, 0.12);
          color: var(--ink-1);
          border: 1px solid rgba(208, 106, 76, 0.22);
          padding: 0.2rem 0.5rem;
          border-radius: 999px;
          font-size: 0.78rem;
          text-decoration: none;
          display: inline-flex;
          align-items: center;
      }}
      .meta-more:hover {{
          border-color: rgba(208, 106, 76, 0.4);
      }}
      .meta-stack {{
          display: grid;
          gap: 0.55rem;
      }}
      .meta-section-title {{
          font-size: 0.68rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--ink-3);
          font-weight: 600;
          margin-bottom: 0.2rem;
      }}
      .meta-note {{
          background: rgba(45, 79, 106, 0.08);
          border: 1px solid rgba(45, 79, 106, 0.16);
          border-radius: 12px;
          padding: 0.45rem 0.6rem;
      }}
      .meta-note-title {{
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--ink-2);
          margin-bottom: 0.2rem;
      }}
      .meta-note-body {{
          font-size: 0.88rem;
          color: var(--ink-1);
      }}
      .meta-rel-group {{
          padding: 0.35rem 0.4rem;
          border-radius: 12px;
          background: rgba(45, 79, 106, 0.05);
          border: 1px solid rgba(45, 79, 106, 0.12);
      }}
      .meta-modal {{
          position: fixed;
          inset: 0;
          background: rgba(15, 23, 42, 0.45);
          display: none;
          align-items: center;
          justify-content: center;
          padding: 1.5rem;
          z-index: 2000;
      }}
      .meta-modal:target {{
          display: flex;
      }}
      .meta-modal-card {{
          background: rgba(255, 255, 255, 0.98);
          border: 1px solid rgba(33, 52, 71, 0.16);
          border-radius: 18px;
          width: min(92vw, 720px);
          max-height: 80vh;
          box-shadow: 0 24px 60px rgba(31, 42, 55, 0.25);
          display: flex;
          flex-direction: column;
          overflow: hidden;
      }}
      .meta-modal-header {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.75rem 1rem;
          border-bottom: 1px solid rgba(33, 52, 71, 0.1);
      }}
      .meta-modal-title {{
          font-family: var(--font-display);
          font-size: 1rem;
          color: var(--ink-1);
      }}
      .meta-modal-close {{
          font-size: 0.8rem;
          color: var(--ink-2);
          text-decoration: none;
          border: 1px solid rgba(33, 52, 71, 0.18);
          padding: 0.2rem 0.55rem;
          border-radius: 999px;
      }}
      .meta-modal-close:hover {{
          border-color: rgba(33, 52, 71, 0.35);
          color: var(--ink-1);
      }}
      .meta-modal-body {{
          padding: 0.85rem 1rem 1.1rem;
          overflow: auto;
      }}
      .meta-modal-list {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
      }}
      .meta-kv-list {{
          display: grid;
          gap: 0.35rem;
      }}
      .meta-kv {{
          display: flex;
          justify-content: space-between;
          gap: 0.5rem;
          font-size: 0.85rem;
          color: var(--ink-2);
      }}
      .meta-kv span {{
          color: var(--ink-3);
      }}
      .meta-kv strong {{
          color: var(--ink-1);
          font-weight: 600;
          text-align: right;
      }}
      .meta-link {{
          color: var(--accent-2);
          text-decoration: none;
          border-bottom: 1px solid rgba(45, 79, 106, 0.35);
      }}
      .meta-link:hover {{
          color: var(--ink-1);
          border-bottom-color: rgba(45, 79, 106, 0.6);
      }}
      .export-page #mynetwork .vis-navigation {{
          bottom: {GRAPH_PANEL_PADDING}px !important;
          left: {GRAPH_PANEL_PADDING}px !important;
          right: {GRAPH_PANEL_PADDING}px !important;
      }}
      .export-page .vis-button {{
          background-color: rgba(255, 255, 255, 0.86);
          border: 1px solid rgba(33, 52, 71, 0.22);
          border-radius: 999px;
          box-shadow: 0 10px 20px rgba(31, 42, 55, 0.12);
          filter: none;
          opacity: 1;
          background-repeat: no-repeat;
          background-position: center !important;
          background-size: 18px 18px !important;
      }}
      .export-page .vis-button:hover {{
          border-color: rgba(33, 52, 71, 0.38);
          box-shadow: 0 12px 22px rgba(31, 42, 55, 0.16);
      }}
      .export-page .vis-button:active {{
          box-shadow: 0 6px 14px rgba(31, 42, 55, 0.12);
      }}
      @media (prefers-reduced-motion: reduce) {{
          * {{
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
              scroll-behavior: auto !important;
          }}
      }}
      @media (max-width: 900px) {{
          body {{
              padding: 18px;
          }}
          .export-title {{
              font-size: 26px;
          }}
          .node-detail-panel {{
              bottom: 12px;
              width: min(520px, 92%);
              max-height: 52%;
          }}
      }}
    </style>
    """


def build_export_html(
    net: Network,
    graph_data: GraphData,
    rel_colors: Dict[str, str],
    node_colors: Dict[str, str],
    export_title: str,
    id_to_label: Optional[Dict[str, str]] = None,
    reduce_motion: bool = False,
    node_animations: Optional[List[str]] = None,
    node_animation_strength: int = 30,
    search_nodes: Optional[List[str]] = None,
    path_nodes: Optional[List[str]] = None,
    centrality_measures: Optional[Dict[str, Dict[str, float]]] = None,
    graph_background: Optional[Dict[str, str]] = None,
    community_colors: Optional[List[str]] = None,
    community_labels: Optional[List[str]] = None,
    show_labels: bool = True,
    smart_labels: bool = False,
    label_zoom_threshold: float = 1.1,
    focus_context: bool = False,
) -> str:
    export_title = export_title.strip() or "Linked Data Explorer"
    safe_title = html_lib.escape(export_title)
    node_total = len(net.nodes)
    edge_total = len(net.edges)
    rel_total = len({edge.get("rel_key") for edge in net.edges if edge.get("rel_key")})
    type_total = (
        len({canonical_type(t) for n in graph_data.nodes for t in n.types}) if graph_data.nodes else 0
    )
    generated_at = time.strftime("%Y-%m-%d %H:%M")
    stats_html = (
        "<div class='export-stats'>"
        f"<div class='export-stat'><span>Nodes</span><strong>{node_total}</strong></div>"
        f"<div class='export-stat'><span>Edges</span><strong>{edge_total}</strong></div>"
        f"<div class='export-stat'><span>Relationships</span><strong>{rel_total}</strong></div>"
        f"<div class='export-stat'><span>Node Types</span><strong>{type_total}</strong></div>"
        "</div>"
    )
    header_html = (
        "<div class='export-hero'>"
        f"<div class='export-title'>{safe_title}</div>"
        f"<div class='export-subtitle'>Network graph export - {generated_at}</div>"
        f"{stats_html}"
        "<div class='export-notes'>Tip: drag to pan, scroll to zoom, click nodes to focus connections.</div>"
        "</div>"
    )
    node_types_html = (
        "<section class='export-legend'>"
        "<div class='export-section-title'>Node Types</div>"
        "<div class='legend-wrap'>"
        f"{create_node_type_legend(node_colors)}"
        "</div>"
        "</section>"
    )
    detail_panel_html = (
        "<div class='node-detail-panel is-empty' id='node-detail-panel'>"
        "<div class='node-detail-title' id='node-detail-title'>Node details</div>"
        "<div class='node-detail-id' id='node-detail-id'></div>"
        "<div class='node-detail-types' id='node-detail-types'></div>"
        "<div class='node-detail-section' id='node-detail-annotation-wrap' style='display:none;'>"
        "<div class='node-detail-section-title'>Annotation</div>"
        "<div class='node-detail-annotation' id='node-detail-annotation'></div>"
        "</div>"
        "<div class='node-detail-section' id='node-detail-props-wrap' style='display:none;'>"
        "<div class='node-detail-section-title'>Metadata</div>"
        "<div class='node-detail-props' id='node-detail-props'></div>"
        "</div>"
        "<div class='node-detail-empty' id='node-detail-empty'>"
        "Click a node to view metadata."
        "</div>"
        "</div>"
    )
    legend_html = (
        "<section class='export-legend'>"
        "<div class='export-section-title'>Legends</div>"
        "<div class='legend-wrap'>"
        f"{create_relationship_legend(rel_colors)}"
        "</div>"
        "</section>"
    )
    community_html = ""
    if getattr(net, "_community_applied", False):
        default_palette = CONFIG.get("COMMUNITY_COLORS") or _COMMUNITY_DEFAULTS
        base_palette = _coerce_color_list(community_colors, list(default_palette))
        community_count = getattr(net, "_community_count", len(base_palette))
        palette = _ensure_palette_size(base_palette, community_count)
        default_labels = [f"Community {idx + 1}" for idx in range(community_count)]
        labels = _coerce_label_list(community_labels, default_labels)
        if len(labels) < community_count:
            labels.extend(default_labels[len(labels) :])
        labels = labels[:community_count]
        if palette:
            community_html = (
                "<section class='export-legend'>"
                "<div class='export-section-title'>Communities</div>"
                "<div class='legend-wrap'>"
                "<div class='legend-card'>"
                "<div class='legend-title'>Communities</div>"
                f"<div class='legend-grid legend-grid-tight'>{create_community_legend(palette, labels)}</div>"
                "</div>"
                "</div>"
                "</section>"
            )
    base_html = net.generate_html()
    base_html = base_html.replace(
        '<div id="mynetwork" class="card-body"></div>',
        '<div class="graph-frame"><div id="mynetwork" class="card-body"></div>'
        + detail_panel_html
        + "</div>",
        1,
    )
    centrality_values = None
    if centrality_measures:
        centrality_values = {
            node_id: float(values.get("degree", 0.0)) for node_id, values in centrality_measures.items()
        }
    label_lookup = refresh_label_index(graph_data, id_to_label)
    metadata_map = _build_export_metadata_map(graph_data, label_lookup=label_lookup)
    label_density_threshold = 60
    custom_js = build_graph_custom_js(
        reduce_motion=reduce_motion,
        node_animations=node_animations,
        node_animation_strength=node_animation_strength,
        search_nodes=search_nodes,
        path_nodes=path_nodes,
        centrality_values=centrality_values,
        graph_background=graph_background,
        metadata_map=metadata_map,
        smart_labels=smart_labels,
        show_labels=show_labels,
        label_zoom_threshold=label_zoom_threshold,
        label_density_threshold=label_density_threshold,
        focus_context=focus_context,
        node_count=node_total,
    )
    combined_css = graph_css_block(
        reduce_motion=reduce_motion, graph_background=graph_background
    ) + export_css_block()
    html = base_html.replace("</head>", f"<title>{safe_title}</title>" + combined_css + "</head>", 1)
    html = html.replace("<body>", "<body><div class='export-page'>" + header_html, 1)
    html = html.replace(
        '<div class="card" style="width: 100%">',
        node_types_html + '<div class="card" style="width: 100%">',
        1,
    )
    legend_block = community_html + legend_html
    html = html.replace("<script type=\"text/javascript\">", legend_block + "<script type=\"text/javascript\">", 1)
    html = html.replace("</body>", custom_js + "</div></body>", 1)
    return html
