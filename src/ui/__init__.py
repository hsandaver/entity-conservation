"""Streamlit application shell."""

from __future__ import annotations

import os


def render_app() -> None:
    import streamlit as st
    
    # CRITICAL: set_page_config() must be called FIRST, before any other streamlit commands
    st.set_page_config(page_title="Linked Data Explorer", layout="wide")
    
    # Now import modules that use streamlit
    from src.config import APP_FONTS
    from src.ui.sidebar import init_session_state, render_sidebar
    from src.ui.tabs import render_tabs
    
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {{
            --bg-0: #FAFAF8;
            --bg-1: #FBF9F6;
            --bg-accent: #F0EAE0;
            --panel: #FFFFFF;
            --panel-muted: #F9F6F1;
            --panel-border: rgba(30, 42, 53, 0.1);
            --ink-1: #0F1419;
            --ink-2: #3A4755;
            --ink-3: #5F6D7D;
            --accent-1: #C85A3A;
            --accent-1-light: #E8897A;
            --accent-2: #1F3F5A;
            --accent-3: #1F8B7E;
            --accent-3-light: #5DB5A8;
            --accent-4: #E5BE6C;
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 8px;
            --shadow-sm: 0 2px 8px rgba(15, 20, 25, 0.06);
            --shadow-md: 0 8px 20px rgba(15, 20, 25, 0.11);
            --shadow-lg: 0 16px 40px rgba(15, 20, 25, 0.14);
            --shadow-xl: 0 24px 48px rgba(15, 20, 25, 0.16);
            --font-display: '{APP_FONTS["display"]}', serif;
            --font-body: '{APP_FONTS["body"]}', sans-serif;
            --font-mono: '{APP_FONTS["mono"]}', monospace;
        }}
        html, body, [class*="css"] {{
            font-family: var(--font-body);
            color: var(--ink-1);
        }}
        body {{
            background-color: var(--bg-0);
        }}
        .stApp {{
            background:
                radial-gradient(ellipse 800px 500px at 5% 10%, rgba(200, 90, 58, 0.1), transparent 60%),
                radial-gradient(ellipse 900px 600px at 95% -5%, rgba(31, 139, 126, 0.08), transparent 55%),
                radial-gradient(circle at 50% 100%, rgba(229, 190, 108, 0.05), transparent 70%),
                linear-gradient(135deg, #FAFAF8 0%, #F0EAE0 50%, #EDE6DD 100%);
        }}
        header[data-testid="stHeader"] {{
            background: linear-gradient(135deg, rgba(250, 250, 248, 0.98), rgba(251, 249, 246, 0.96));
            border-bottom: 1px solid rgba(30, 42, 53, 0.08);
            box-shadow: var(--shadow-sm);
            backdrop-filter: blur(10px);
        }}
        header[data-testid="stHeader"] * {{
            color: var(--ink-2) !important;
        }}
        header[data-testid="stHeader"] svg {{
            fill: var(--ink-2) !important;
        }}
        header[data-testid="stHeader"] a {{
            color: var(--ink-2) !important;
        }}
        div[data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        .block-container {{
            padding-top: 3rem;
        }}
        h1, h2, h3, h4, h5, h6,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] h5,
        div[data-testid="stMarkdownContainer"] h6 {{
            font-family: var(--font-display);
            color: var(--ink-1);
            letter-spacing: -0.3px;
            font-feature-settings: "ss01", "ss02";
        }}
        h1 {{
            font-weight: 700;
            font-size: 2.2rem;
            line-height: 1.2;
        }}
        h2 {{
            font-weight: 650;
            font-size: 1.8rem;
            line-height: 1.3;
        }}
        h3 {{
            font-weight: 600;
            font-size: 1.4rem;
            line-height: 1.4;
        }}
        p, li, label, span,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {{
            color: var(--ink-1);
            line-height: 1.6;
        }}
        code, pre {{
            font-family: var(--font-mono);
            background: rgba(15, 20, 25, 0.04);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            color: var(--accent-2);
        }}
        pre {{
            padding: 1rem;
            background: rgba(15, 20, 25, 0.06);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            overflow-x: auto;
        }}
        @keyframes riseFade {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes chipFade {{
            from {{ opacity: 0; transform: translateY(8px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: 0% 0%; }}
            100% {{ background-position: 100% 0%; }}
        }}
        @keyframes glowPulse {{
            0%, 100% {{ box-shadow: var(--shadow-md); }}
            50% {{ box-shadow: 0 12px 32px rgba(200, 90, 58, 0.2); }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }}
            .hero,
            .hero-chips span {{
                animation: none !important;
            }}
        }}
        section[data-testid="stSidebar"], div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(251, 249, 246, 0.85) 0%, rgba(240, 234, 224, 0.85) 100%);
            border-right: 1px solid rgba(30, 42, 53, 0.08);
            backdrop-filter: blur(8px);
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 1.5rem;
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--ink-1);
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(249, 246, 241, 0.95) 100%);
            border-radius: 16px;
            border: 1px solid rgba(30, 42, 53, 0.1);
            box-shadow: var(--shadow-sm);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details:hover {{
            box-shadow: var(--shadow-md);
            border-color: rgba(200, 90, 58, 0.2);
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
            padding: 0.6rem 0.8rem;
            font-weight: 650;
            color: var(--ink-1);
            cursor: pointer;
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg {{
            color: var(--accent-1);
        }}
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
            padding: 0.6rem 0.4rem 0.8rem 0.4rem;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(30, 42, 53, 0.08);
        }}
        .stButton > button {{
            background: linear-gradient(135deg, var(--accent-1), #E07055);
            color: #FFFFFF;
            border: 1px solid rgba(200, 90, 58, 0.3);
            border-radius: 999px;
            padding: 0.65rem 1.4rem;
            font-weight: 700;
            font-size: 0.95rem;
            box-shadow: var(--shadow-md), 0 8px 16px rgba(200, 90, 58, 0.15);
            transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
            letter-spacing: 0.3px;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg), 0 12px 24px rgba(200, 90, 58, 0.25);
        }}
        .stButton > button:active {{
            transform: translateY(0px);
        }}
        .stDownloadButton > button,
        div[data-testid="stDownloadButton"] > button {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 246, 241, 0.95));
            color: var(--ink-1);
            border: 1.5px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.65rem 1.4rem;
            font-weight: 700;
            box-shadow: var(--shadow-md);
            transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
            letter-spacing: 0.3px;
            outline: none;
            position: relative;
            z-index: 1;
        }}
        .stDownloadButton,
        div[data-testid="stDownloadButton"] {{
            overflow: visible;
            margin-bottom: 0.5rem;
        }}
        .stDownloadButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            transform: translateY(-2px);
            border-color: rgba(200, 90, 58, 0.5);
            box-shadow:
                var(--shadow-lg),
                inset 0 0 0 1px rgba(200, 90, 58, 0.25);
            background: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(250, 247, 242, 0.98));
        }}
        .stDownloadButton > button:focus,
        div[data-testid="stDownloadButton"] > button:focus {{
            outline: none;
            box-shadow: var(--shadow-md);
        }}
        .stDownloadButton > button:focus:not(:focus-visible),
        div[data-testid="stDownloadButton"] > button:focus:not(:focus-visible) {{
            border-color: var(--panel-border);
        }}
        .stDownloadButton > button:focus-visible,
        div[data-testid="stDownloadButton"] > button:focus-visible {{
            border-color: rgba(200, 90, 58, 0.7);
            box-shadow:
                var(--shadow-lg),
                inset 0 0 0 2px rgba(200, 90, 58, 0.35);
        }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1.5px solid rgba(30, 42, 53, 0.15) !important;
            border-radius: 14px !important;
            box-shadow: inset 0 1px 3px rgba(15, 20, 25, 0.05) !important;
            transition: all 0.3s ease;
        }}
        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="textarea"] > div:hover,
        div[data-baseweb="select"] > div:hover {{
            border-color: rgba(30, 42, 53, 0.25) !important;
            box-shadow: inset 0 1px 3px rgba(15, 20, 25, 0.08) !important;
        }}
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: var(--accent-1) !important;
            box-shadow: inset 0 1px 3px rgba(15, 20, 25, 0.05), 0 0 0 3px rgba(200, 90, 58, 0.1) !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {{
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
            color: var(--ink-1) !important;
            padding: 0.65rem 0.85rem !important;
            font-weight: 500;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="textarea"] textarea {{
            min-height: 120px;
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
            border-radius: 999px;
            background: rgba(30, 42, 53, 0.12);
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {{
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-3));
            box-shadow: 0 2px 8px rgba(200, 90, 58, 0.2);
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
            background: #FFFFFF;
            border: 2.5px solid var(--accent-1);
            box-shadow: var(--shadow-md);
            transition: all 0.2s ease;
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:hover {{
            border-color: var(--accent-1);
            box-shadow: var(--shadow-lg);
            transform: scale(1.1);
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"]:focus-visible {{
            outline: none;
            border-color: var(--accent-3);
            box-shadow:
                0 0 0 4px rgba(31, 139, 126, 0.2),
                var(--shadow-lg);
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] span {{
            background: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child,
        div[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child div,
        div[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child span {{
            background: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            color: var(--ink-2) !important;
        }}
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.12), rgba(224, 112, 85, 0.08)) !important;
            color: var(--accent-1) !important;
            border: 1.5px solid rgba(200, 90, 58, 0.35) !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            font-weight: 700;
            transition: all 0.2s ease;
        }}
        div[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.18), rgba(224, 112, 85, 0.12)) !important;
            border-color: rgba(200, 90, 58, 0.5) !important;
        }}
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{
            color: var(--accent-1) !important;
        }}
        div[data-testid="stMultiSelect"] [data-baseweb="select"] button {{
            background: transparent !important;
            box-shadow: none !important;
        }}
        div[data-testid="stMultiSelect"] [data-baseweb="select"] svg {{
            color: var(--accent-1) !important;
        }}
        div[data-testid="stDeckGlChart"] {{
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(30, 42, 53, 0.1);
            background: #FFFFFF;
            box-shadow: var(--shadow-lg);
        }}
        div[data-testid="stDeckGlChart"] canvas {{
            border-radius: 16px;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-top-right,
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-top-left {{
            margin: 0.65rem;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-group {{
            border: 1px solid rgba(33, 52, 71, 0.16);
            border-radius: 12px;
            box-shadow: 0 10px 18px rgba(31, 42, 55, 0.12);
            overflow: hidden;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-group button {{
            width: 32px;
            height: 32px;
            background-color: #FFFFFF !important;
            border: none !important;
            background-repeat: no-repeat;
            background-position: center;
            background-size: 16px 16px;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-group button.mapboxgl-ctrl-zoom-in {{
            background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%2712%27%20y1%3D%275%27%20x2%3D%2712%27%20y2%3D%2719%27/%3E%3Cline%20x1%3D%275%27%20y1%3D%2712%27%20x2%3D%2719%27%20y2%3D%2712%27/%3E%3C/svg%3E") !important;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-group button.mapboxgl-ctrl-zoom-out {{
            background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%275%27%20y1%3D%2712%27%20x2%3D%2719%27%20y2%3D%2712%27/%3E%3C/svg%3E") !important;
        }}
        div[data-testid="stDeckGlChart"] .mapboxgl-ctrl-group .mapboxgl-ctrl-icon {{
            background-image: none !important;
        }}
        .mapboxgl-ctrl-group button {{
            width: 32px;
            height: 32px;
            background-color: #FFFFFF !important;
            border: none !important;
            background-repeat: no-repeat;
            background-position: center;
            background-size: 16px 16px;
        }}
        .mapboxgl-ctrl-group button.mapboxgl-ctrl-zoom-in {{
            background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%2712%27%20y1%3D%275%27%20x2%3D%2712%27%20y2%3D%2719%27/%3E%3Cline%20x1%3D%275%27%20y1%3D%2712%27%20x2%3D%2719%27%20y2%3D%2712%27/%3E%3C/svg%3E") !important;
        }}
        .mapboxgl-ctrl-group button.mapboxgl-ctrl-zoom-out {{
            background-image: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%232D3748%27%20stroke-width%3D%272%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cline%20x1%3D%275%27%20y1%3D%2712%27%20x2%3D%2719%27%20y2%3D%2712%27/%3E%3C/svg%3E") !important;
        }}
        .mapboxgl-ctrl-group .mapboxgl-ctrl-icon {{
            background-image: none !important;
        }}
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: rgba(45, 79, 106, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(45, 79, 106, 0.15) !important;
            background: #FFFFFF !important;
        }}
        div[data-baseweb="input"][data-invalid="true"] > div,
        div[data-baseweb="textarea"][data-invalid="true"] > div,
        div[data-baseweb="select"][data-invalid="true"] > div {{
            border-color: rgba(33, 52, 71, 0.2) !important;
            box-shadow: inset 0 1px 2px rgba(31, 42, 55, 0.08) !important;
        }}
        div[data-baseweb="input"][data-invalid="true"] > div:focus-within,
        div[data-baseweb="textarea"][data-invalid="true"] > div:focus-within,
        div[data-baseweb="select"][data-invalid="true"] > div:focus-within {{
            border-color: rgba(45, 79, 106, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(45, 79, 106, 0.15) !important;
        }}
        div[data-baseweb="input"] input:invalid,
        div[data-baseweb="textarea"] textarea:invalid {{
            outline: none !important;
            box-shadow: none !important;
        }}
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {{
            color: var(--ink-3);
        }}
        label[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"],
        div[data-testid="stCheckbox"] label {{
            color: var(--ink-1);
        }}
        label[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] {{
            display: flex;
            align-items: center;
            gap: 0.35rem;
            justify-content: flex-start !important;
        }}
        label[data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
        div[data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] {{
            margin-left: 0.2rem !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"] {{
            border: 1px solid rgba(33, 52, 71, 0.35) !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="checkbox"] div[role="checkbox"] {{
            border: 1px solid rgba(33, 52, 71, 0.35) !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] [role="checkbox"] {{
            border: 1px solid rgba(33, 52, 71, 0.35) !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {{
            background: var(--accent-2) !important;
            border-color: var(--accent-2) !important;
        }}
        div[data-baseweb="checkbox"] div[role="checkbox"][aria-checked="true"] {{
            background: var(--accent-2) !important;
            border-color: var(--accent-2) !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"] svg {{
            color: #FFFFFF !important;
        }}
        div[data-baseweb="checkbox"] svg {{
            color: #FFFFFF !important;
        }}
        div[data-testid="stFileUploaderDropzone"] {{
            background: var(--panel);
            border: 1px dashed rgba(45, 79, 106, 0.32) !important;
            border-radius: 14px;
            color: var(--ink-2);
            box-shadow: none !important;
        }}
        div[data-testid="stFileUploaderDropzone"] > div,
        div[data-testid="stFileUploaderDropzone"] section {{
            background: #FFFFFF !important;
        }}
        div[data-testid="stFileUploader"] section {{
            border: 1px solid rgba(33, 52, 71, 0.14) !important;
            border-radius: 16px;
            background: var(--panel);
            box-shadow: 0 10px 20px rgba(31, 42, 55, 0.06);
        }}
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {{
            background: #FFFFFF;
            border-color: rgba(33, 52, 71, 0.16) !important;
        }}
        div[data-testid="stFileUploaderDropzone"] svg {{
            color: var(--accent-2);
        }}
        div[data-testid="stFileUploader"] button {{
            background: var(--panel);
            color: var(--ink-1);
            border-radius: 999px;
            border: 1px solid rgba(33, 52, 71, 0.18);
            box-shadow: 0 8px 16px rgba(31, 42, 55, 0.08);
        }}
        div[data-testid="stExpander"] details {{
            background: var(--panel);
            border-radius: var(--radius-lg);
            border: 1px solid var(--panel-border);
            padding: 0.35rem 0.75rem;
            box-shadow: var(--shadow-soft);
            overflow: visible;
        }}
        div[data-testid="stExpander"] summary {{
            font-weight: 600;
            color: var(--ink-1);
        }}
        div[data-testid="stTabs"] div[role="tablist"] {{
            background: rgba(255, 255, 255, 0.92) !important;
            border-radius: 999px !important;
            padding: 0.35rem !important;
            gap: 0.35rem !important;
            border: 1px solid rgba(33, 52, 71, 0.16) !important;
            box-shadow: 0 10px 24px rgba(31, 42, 55, 0.08) !important;
            overflow-x: auto !important;
            scrollbar-width: none !important;
        }}
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {{
            display: none !important;
            background: none !important;
            border: none !important;
        }}
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            display: none !important;
            background: none !important;
            border: none !important;
        }}
        div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {{
            height: 0 !important;
        }}
        div[data-testid="stTabs"] button {{
            border-radius: 999px !important;
            padding: 0.45rem 1rem !important;
            color: var(--ink-2) !important;
            font-weight: 600 !important;
            border: 1px solid transparent !important;
            border-bottom: none !important;
            flex: 0 0 auto !important;
            background: transparent !important;
        }}
        div[data-testid="stTabs"] button:hover {{
            background: rgba(255, 255, 255, 0.5) !important;
        }}
        div[data-testid="stTabs"] button::after {{
            display: none !important;
        }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{
            background: var(--panel) !important;
            color: var(--ink-1) !important;
            border-color: rgba(33, 52, 71, 0.18) !important;
            box-shadow: 0 10px 20px rgba(31, 42, 55, 0.12) !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stForm"] {{
            overflow: visible;
        }}
        div[data-testid="stMetric"] {{
            background: var(--panel);
            border-radius: var(--radius-md);
            border: 1px solid rgba(33, 52, 71, 0.14);
            padding: 0.75rem 0.9rem;
            box-shadow: 0 12px 24px rgba(31, 42, 55, 0.08);
        }}
        div[data-testid="stMetric"] label {{
            color: var(--ink-3);
        }}
        div[data-testid="stAlert"] {{
            border-radius: var(--radius-md);
            border: 1px solid rgba(33, 52, 71, 0.2);
            box-shadow: 0 10px 20px rgba(31, 42, 55, 0.08);
        }}
        div[data-testid="stToolbar"] {{
            color: var(--ink-2);
        }}
        div[data-testid="stToolbar"] button,
        div[data-testid="stToolbar"] svg {{
            color: var(--ink-2) !important;
        }}
        div[data-testid="stToolbar"] div[role="menu"] {{
            background: #FFFFFF !important;
            border: 1px solid rgba(33, 52, 71, 0.16) !important;
            border-radius: 12px;
            box-shadow: 0 16px 28px rgba(31, 42, 55, 0.16);
        }}
        div[data-testid="stToolbar"] div[role="menu"] div[role="menuitem"] {{
            color: var(--ink-1) !important;
        }}
        iframe[title="streamlit.components.v1.html"] {{
            border-radius: var(--radius-lg);
            border: none;
            box-shadow: none;
            background: transparent;
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 255, 255, 0.92) 50%, rgba(248, 244, 237, 0.94) 100%);
            border: 1px solid rgba(30, 42, 53, 0.12);
            border-radius: 28px;
            padding: 2.4rem 2.8rem;
            box-shadow: 0 24px 48px rgba(15, 20, 25, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.7);
            margin-bottom: 2rem;
            animation: riseFade 0.8s ease both;
            position: relative;
            overflow: hidden;
        }}
        .hero::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.8), transparent);
            pointer-events: none;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            bottom: -100%;
            right: -10%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(200, 90, 58, 0.05), transparent 70%);
            pointer-events: none;
            border-radius: 50%;
        }}
        .hero-title {{
            font-family: var(--font-display);
            font-size: 2.8rem;
            font-weight: 750;
            color: var(--ink-1);
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.5px;
            line-height: 1.1;
            background: linear-gradient(135deg, var(--ink-1) 0%, var(--ink-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hero-subtitle {{
            font-size: 1.15rem;
            color: var(--ink-2);
            max-width: 65rem;
            margin-bottom: 1.3rem;
            line-height: 1.7;
            font-weight: 500;
        }}
        .hero-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }}
        .hero-chips span {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.08) 0%, rgba(31, 139, 126, 0.06) 100%);
            color: var(--ink-1);
            border: 1.5px solid rgba(30, 42, 53, 0.15);
            padding: 0.45rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            opacity: 0;
            animation: chipFade 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
        }}
        .hero-chips span:hover {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.15) 0%, rgba(31, 139, 126, 0.12) 100%);
            border-color: rgba(200, 90, 58, 0.35);
        }}
        .hero-chips span:nth-child(1) {{ animation-delay: 0.08s; }}
        .hero-chips span:nth-child(2) {{ animation-delay: 0.16s; }}
        .hero-chips span:nth-child(3) {{ animation-delay: 0.24s; }}
        .hero-chips span:nth-child(4) {{ animation-delay: 0.32s; }}
        .hero-chips span:nth-child(5) {{ animation-delay: 0.4s; }}
        .hero-chips span:nth-child(6) {{ animation-delay: 0.48s; }}
        .legend-card {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 244, 237, 0.94) 100%);
            border: 1px solid rgba(30, 42, 53, 0.1);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: var(--shadow-md);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        .legend-card:hover {{
            box-shadow: var(--shadow-lg);
            border-color: rgba(200, 90, 58, 0.3);
        }}
        .legend-title {{
            font-weight: 750;
            margin-bottom: 0.8rem;
            color: var(--ink-1);
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.5rem 0.75rem;
        }}
        .legend-grid-tight {{
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--ink-2);
            font-size: 0.95rem;
            font-weight: 500;
        }}
        .legend-swatch {{
            width: 12px;
            height: 12px;
            border-radius: 999px;
            border: 1.5px solid rgba(0, 0, 0, 0.12);
            display: inline-block;
            flex-shrink: 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        }}
        .meta-panel {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.97) 0%, rgba(248, 244, 237, 0.95) 100%);
            border: 1px solid rgba(30, 42, 53, 0.1);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            box-shadow: var(--shadow-md);
            margin-top: 0.8rem;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        .meta-panel:hover {{
            box-shadow: var(--shadow-lg);
        }}
        .annotation-preview {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.97) 0%, rgba(248, 244, 237, 0.95) 100%);
            border: 1px solid rgba(30, 42, 53, 0.1);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: var(--shadow-md);
            margin-bottom: 0.9rem;
            transition: all 0.3s ease;
        }}
        .annotation-preview:hover {{
            box-shadow: var(--shadow-lg);
        }}
        .annotation-preview p {{
            margin: 0 0 0.5rem 0;
            color: var(--ink-1);
            line-height: 1.6;
        }}
        .annotation-preview p:last-child {{
            margin-bottom: 0;
        }}
        .annotation-preview ul,
        .annotation-preview ol {{
            margin: 0.5rem 0;
            padding-left: 1.3rem;
            color: var(--ink-1);
        }}
        .annotation-preview a {{
            color: var(--accent-2);
            text-decoration: none;
            border-bottom: 1.5px solid rgba(31, 63, 90, 0.3);
            transition: all 0.2s ease;
        }}
        .annotation-preview a:hover {{
            border-bottom-color: var(--accent-2);
        }}
        .meta-header {{
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--ink-1);
            margin-bottom: 1rem;
            letter-spacing: -0.2px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}
        .meta-card {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.99) 0%, rgba(248, 244, 237, 0.96) 100%);
            border: 1px solid rgba(30, 42, 53, 0.1);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow-md);
            min-height: 96px;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        .meta-card:hover {{
            box-shadow: var(--shadow-lg), 0 12px 24px rgba(200, 90, 58, 0.12);
            border-color: rgba(200, 90, 58, 0.3);
        }}
        .meta-card-more {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 234, 224, 0.94) 100%);
        }}
        .meta-key {{
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink-3);
            font-weight: 700;
            margin-bottom: 0.45rem;
        }}
        .meta-value {{
            font-size: 1rem;
            color: var(--ink-1);
            line-height: 1.5;
            word-break: break-word;
            font-weight: 500;
        }}
        .meta-muted {{
            color: var(--ink-3);
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .meta-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .meta-chip {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.1) 0%, rgba(31, 139, 126, 0.08) 100%);
            color: var(--ink-1);
            border: 1.5px solid rgba(30, 42, 53, 0.12);
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        .meta-chip:hover {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.15) 0%, rgba(31, 139, 126, 0.12) 100%);
            border-color: rgba(200, 90, 58, 0.3);
        }}
            text-decoration: none;
            line-height: 1.2;
        }}
        .meta-chip:hover {{
            border-color: rgba(33, 52, 71, 0.28);
        }}
        .meta-chip-primary {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.12) 0%, rgba(224, 112, 85, 0.08) 100%);
            border-color: rgba(200, 90, 58, 0.3);
        }}
        .meta-more {{
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.1) 0%, rgba(31, 139, 126, 0.08) 100%);
            color: var(--accent-1);
            border: 1.5px solid rgba(200, 90, 58, 0.25);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.85rem;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            font-weight: 700;
            transition: all 0.2s ease;
        }}
        .meta-more:hover {{
            border-color: rgba(200, 90, 58, 0.5);
            background: linear-gradient(135deg, rgba(200, 90, 58, 0.18) 0%, rgba(31, 139, 126, 0.12) 100%);
        }}
        .meta-stack {{
            display: grid;
            gap: 0.7rem;
        }}
        .meta-section-title {{
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--ink-3);
            font-weight: 750;
            margin-bottom: 0.3rem;
        }}
        .meta-note {{
            background: linear-gradient(135deg, rgba(45, 79, 106, 0.08) 0%, rgba(31, 139, 126, 0.05) 100%);
            border: 1.5px solid rgba(30, 42, 53, 0.12);
            border-radius: 14px;
            padding: 0.65rem 0.85rem;
            transition: all 0.2s ease;
        }}
        .meta-note:hover {{
            background: linear-gradient(135deg, rgba(45, 79, 106, 0.12) 0%, rgba(31, 139, 126, 0.08) 100%);
            box-shadow: var(--shadow-sm);
        }}
        .meta-note-title {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-2);
            margin-bottom: 0.3rem;
        }}
        .meta-note-body {{
            font-size: 0.92rem;
            color: var(--ink-1);
            line-height: 1.5;
        }}
        .meta-rel-group {{
            padding: 0.5rem 0.6rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(30, 42, 53, 0.04) 0%, rgba(200, 90, 58, 0.03) 100%);
            border: 1.5px solid rgba(30, 42, 53, 0.1);
            transition: all 0.2s ease;
        }}
        .meta-rel-group:hover {{
            background: linear-gradient(135deg, rgba(30, 42, 53, 0.08) 0%, rgba(200, 90, 58, 0.06) 100%);
            box-shadow: var(--shadow-sm);
        }}
        .meta-modal {{
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.55);
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            z-index: 1000005;
            backdrop-filter: blur(4px);
        }}
        .meta-modal:target {{
            display: flex;
            animation: riseFade 0.3s ease both;
        }}
        .meta-modal-card {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.99) 0%, rgba(248, 244, 237, 0.98) 100%);
            border: 1px solid rgba(30, 42, 53, 0.12);
            border-radius: 20px;
            width: 720px;
            max-width: 94vw;
            max-height: 88vh;
            box-shadow: 0 40px 80px rgba(15, 20, 25, 0.22), 0 0 0 1px rgba(30, 42, 53, 0.04);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            pointer-events: auto;
            transform: translateZ(0);
        }}
        .meta-modal-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.4rem;
            border-bottom: 1.5px solid rgba(30, 42, 53, 0.08);
            background: linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 244, 237, 0.8) 100%);
        }}
        .meta-modal-title {{
            font-family: var(--font-display);
            font-size: 1.2rem;
            font-weight: 750;
            color: var(--ink-1);
            letter-spacing: -0.2px;
        }}
        .meta-modal-close {{
            font-size: 0.9rem;
            color: var(--ink-2);
            text-decoration: none;
            border: 1.5px solid rgba(30, 42, 53, 0.15);
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        .meta-modal-close:hover {{
            border-color: rgba(200, 90, 58, 0.4);
            color: var(--accent-1);
            background: rgba(200, 90, 58, 0.06);
        }}
        .meta-modal-body {{
            padding: 1.2rem 1.4rem;
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
        /* Chromium tablist right-edge spacer to prevent rounded shadow clipping */
        div[data-testid="stTabs"] div[role="tablist"] {{
            padding-right: 1.5rem !important;
            scrollbar-gutter: stable both-edges;
        }}
        div[data-testid="stTabs"] div[role="tablist"]::after {{
            content: "" !important;
            display: block;
            width: 1.5rem;
            flex: 0 0 1.5rem;
            height: 1px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Linked Data Explorer</div>
            <div class="hero-subtitle">
                Visualize, analyze, and curate linked data with a research-grade network studio for RDF, SPARQL,
                embeddings, and semantic enrichment.
            </div>
            <div class="hero-chips">
                <span>RDF + JSON-LD</span>
                <span>SPARQL</span>
                <span>Embeddings</span>
                <span>IIIF</span>
                <span>Semantic Analysis</span>
                <span>Graph Editing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Inject runtime JS spacer (ensures tablist padding even if CSS ordering differs on hosts)
    try:
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "tab_spacer.js")
        with open(asset_path, "r", encoding="utf-8") as _f:
            _js = _f.read()
        st.markdown(f"<script>{_js}</script>", unsafe_allow_html=True)
    except Exception:
        pass

    with st.expander("Quick Start"):
        st.write("1. Upload your JSON/JSON-LD/RDF files in the **File Upload** section on the sidebar.")
        st.write("2. Optionally, upload a SHACL shapes file to validate your data.")
        st.write("3. Adjust visualization settings like physics, filtering, and centrality measures.")
        st.write("4. (Optional) Run SPARQL queries or load data from a remote SPARQL endpoint.")
        st.write("5. Use URI Dereferencing to load external RDF data by providing a list of URIs.")
        st.write("6. Explore the graph in the **Graph View** tab below!")
        st.write("7. Set manual node positions using the sidebar.")

    init_session_state()

    if st.session_state.reduce_motion:
        st.markdown(
            """
            <style>
            * {
                animation: none !important;
                transition: none !important;
                scroll-behavior: auto !important;
            }
            .hero,
            .hero-chips span {
                animation: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    sidebar_state = render_sidebar()
    render_tabs(
        filtered_nodes=sidebar_state.filtered_nodes,
        community_detection=sidebar_state.community_detection,
        effective_node_animations=sidebar_state.effective_node_animations,
    )
