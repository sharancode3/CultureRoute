"""CultureRoute Editorial UI Presentation Components.

Implements the editorial design system matching the demo prototype (site/dist):
- Palette: Paper #f7f3e9, Charcoal Ink #292925, Oxblood #982b32, Quiet Blue #365376,
  Muted Ink #5c5852 (WCAG 2.2 AA compliant contrast > 4.5:1), Pale Yellow Note #eee3a6.
- Typography: Hand-lettered display headings ("Segoe Print", "Comic Sans MS", "Trebuchet MS", cursive),
  editorial body ("Trebuchet MS", "Segoe UI", sans-serif).
- Dot-matrix tactile paper overlay texture.
- Hand-drawn SVG accents: wavy hero underline, 16-point doodle star, looping indicator arrow.
- Photographed cards with white framing borders, lifted layout, and golden glow hover transitions.
- Tactile pressed buttons, taped sticky notes, and styled editorial navigation.
- Real semantic link navigation with aligned local SVG line icons, grouped by Explore, Study, Help.
- Full reduced-motion compliance honoring both OS settings and in-app preferences.
- Zero emojis and zero unverified real-world travel claims.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
import plotly.io as pio
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "site" / "dist" / "assets"


@lru_cache(maxsize=1)
def configure_editorial_plotly_theme() -> None:
    """Register and activate a light warm-paper Plotly template immune to OS dark mode."""
    base_tmpl = pio.templates["plotly_white"]
    custom_tmpl = base_tmpl.to_plotly_json()
    layout = custom_tmpl.setdefault("layout", {})
    layout["paper_bgcolor"] = "#f7f3e9"
    layout["plot_bgcolor"] = "#ffffff"
    layout["font"] = {"color": "#292925", "family": "Trebuchet MS, Segoe UI, sans-serif", "size": 12}
    layout["title"] = {"font": {"color": "#292925", "size": 15}}
    layout["legend"] = {"font": {"color": "#292925", "size": 11}, "bgcolor": "rgba(255,255,255,0)"}
    layout["hoverlabel"] = {
        "bgcolor": "#ffffff",
        "bordercolor": "#b6afa3",
        "font": {"color": "#292925", "family": "Trebuchet MS, Segoe UI, sans-serif", "size": 12},
    }
    axis_style = {
        "color": "#292925",
        "gridcolor": "#ede8dc",
        "linecolor": "#b6afa3",
        "zerolinecolor": "#d5cebf",
        "tickfont": {"color": "#5c5852", "size": 11},
        "title": {"font": {"color": "#292925", "size": 12}},
    }
    layout["xaxis"] = {**layout.get("xaxis", {}), **axis_style}
    layout["yaxis"] = {**layout.get("yaxis", {}), **axis_style}
    pio.templates["cultureroute_paper"] = custom_tmpl
    pio.templates.default = "cultureroute_paper"


def apply_editorial_chart_theme(fig):
    """Explicitly enforce warm-paper light font/hover/axis colors on any Plotly figure."""
    fig.update_layout(
        font=dict(color="#292925", family="Trebuchet MS, Segoe UI, sans-serif"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#b6afa3", font=dict(color="#292925")),
        legend=dict(font=dict(color="#292925")),
    )
    fig.update_xaxes(tickfont=dict(color="#5c5852"), title_font=dict(color="#292925"), gridcolor="#ede8dc", linecolor="#b6afa3")
    fig.update_yaxes(tickfont=dict(color="#5c5852"), title_font=dict(color="#292925"), gridcolor="#ede8dc", linecolor="#b6afa3")
    return fig


@lru_cache(maxsize=16)
def get_asset_base64(filename: str) -> str:
    """Read an image file from site/dist/assets and return data URL."""
    filepath = ASSETS_DIR / filename
    if not filepath.exists():
        return ""
    data = filepath.read_bytes()
    suffix = filepath.suffix.lower().lstrip(".")
    if suffix in ["jpg", "jpeg"]:
        mime = "image/jpeg"
    elif suffix == "avif":
        mime = "image/avif"
    else:
        mime = f"image/{suffix}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def inject_editorial_css() -> None:
    """Inject global editorial stylesheet matching site/dist prototype into Streamlit."""
    configure_editorial_plotly_theme()
    is_reduced = st.session_state.get("motion_preference") == "Reduced Motion (Accessibility)"
    st.markdown(_build_editorial_css(is_reduced), unsafe_allow_html=True)


@lru_cache(maxsize=2)
def _build_editorial_css(is_reduced: bool) -> str:
    reduced_motion_css = """
    *, *:before, *:after {
        transition: none !important;
        animation: none !important;
        transform: none !important;
    }
    """ if is_reduced else ""

    css = f"""
<style>
    /* CSS Design Tokens matching prototype */
    :root {{
        color-scheme: light !important;
        --paper: #f7f3e9;
        --ink: #292925;
        --red: #982b32;
        --oxblood: #982b32;
        --muted: #5c5852;
        --muted-ink: #5c5852;
        --line: #b6afa3;
        --border-line: #b6afa3;
        --yellow: #eee3a6;
        --pale-yellow: #eee3a6;
        --blue: #365376;
        --quiet-blue: #365376;
        --panel: #eee9dd;
        --panel-bg: #eee9dd;
        --card-white: #ffffff;
        --display: "Segoe Print", "Comic Sans MS", "Trebuchet MS", cursive;
        --body: "Trebuchet MS", "Segoe UI", sans-serif;
    }}

    /* Base Paper Canvas */
    html {{
        scroll-behavior: smooth;
        color-scheme: light !important;
    }}
    body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: var(--paper) !important;
        color: var(--ink) !important;
        font-family: var(--body) !important;
        font-size: 16px;
        line-height: 1.55;
        color-scheme: light !important;
    }}

    /* Hide Deploy button and Streamlit menu clutter while preserving sidebar collapse and expand buttons */
    .stDeployButton,
    [data-testid="stDeployButton"],
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    button[data-testid="stAppDeployButton"],
    .stToolbarActions,
    [data-testid="stToolbarActions"],
    [data-testid="stHeaderActionElements"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
    }}

    header[data-testid="stHeader"],
    [data-testid="stHeader"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        z-index: 99999 !important;
    }}

    /* =========================================================================
       SIDEBAR THEME & CONTROLS: COLLAPSIBLE MINI-RAIL & EXPANDED VIEWS
       ========================================================================= */

    /* Completely eliminate sidebar scrolling and hide scrollbars */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        overflow: hidden !important;
        overflow-y: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }}

    [data-testid="stSidebarContent"]::-webkit-scrollbar,
    [data-testid="stSidebarUserContent"]::-webkit-scrollbar,
    section[data-testid="stSidebar"]::-webkit-scrollbar {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }}

    /* Eliminate Streamlit's default 96px bottom padding on sidebar */
    [data-testid="stSidebarUserContent"] {{
        padding-bottom: 8px !important;
        padding-top: 0 !important;
    }}

    /* Hide empty logo spacer & shrink header to prevent height waste */
    [data-testid="stLogoSpacer"] {{
        display: none !important;
    }}
    [data-testid="stSidebarHeader"] {{
        padding: 4px 10px 0 !important;
        min-height: 0 !important;
        height: 24px !important;
        margin-bottom: 2px !important;
    }}

    /* Expanded Sidebar: 260px width with rich branding and full navigation */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {{
        background-color: #f0ebe0 !important;
        border-right: 1.5px solid var(--line) !important;
        box-shadow: 2px 0 12px rgba(41, 41, 37, 0.06) !important;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        position: relative !important;
        transform: none !important;
        transition: width 0.22s cubic-bezier(0.16, 1, 0.3, 1), min-width 0.22s cubic-bezier(0.16, 1, 0.3, 1), max-width 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    /* Collapsed Sidebar: slim 68px icon rail keeping SVGs visible & interactive */
    [data-testid="stSidebar"][data-collapsed="true"],
    section[data-testid="stSidebar"][data-collapsed="true"] {{
        width: 68px !important;
        min-width: 68px !important;
        max-width: 68px !important;
        transform: none !important;
        margin-left: 0 !important;
        overflow-x: hidden !important;
        background-color: #f0ebe0 !important;
        border-right: 1.5px solid var(--line) !important;
        box-shadow: 2px 0 8px rgba(41, 41, 37, 0.06) !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarContent"] {{
        width: 68px !important;
        min-width: 68px !important;
        max-width: 68px !important;
        padding: 14px 4px !important;
        overflow-x: hidden !important;
    }}

    /* Rail mode elements visibility */
    .cr-brand-mini {{
        display: none;
    }}
    .cr-brand-full {{
        display: inline;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-brand-full,
    [data-testid="stSidebar"][data-collapsed="true"] .cr-brand-subtitle,
    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-group-title,
    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-link span,
    [data-testid="stSidebar"][data-collapsed="true"] .sticky {{
        display: none !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-brand-mini {{
        display: inline-block !important;
        font-family: var(--display) !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        color: var(--red) !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-sidebar-brand {{
        padding: 4px 0 10px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid var(--line) !important;
        text-align: center !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-group {{
        margin-bottom: 6px !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-link,
    [data-testid="stSidebar"][data-collapsed="true"] a.cr-nav-link {{
        width: 44px !important;
        height: 44px !important;
        margin: 4px auto !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px !important;
        transform: none !important;
        border: 1px solid transparent !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-link svg,
    [data-testid="stSidebar"][data-collapsed="true"] a.cr-nav-link svg {{
        margin: 0 !important;
        width: 20px !important;
        height: 20px !important;
        transform: none !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] a.cr-nav-link.is-active,
    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-link.is-active {{
        background-color: var(--yellow) !important;
        border: 1.5px solid var(--red) !important;
        box-shadow: 0 2px 5px rgba(50, 43, 28, 0.15) !important;
        transform: none !important;
    }}

    [data-testid="stSidebar"][data-collapsed="true"] a.cr-nav-link:hover,
    [data-testid="stSidebar"][data-collapsed="true"] .cr-nav-link:hover {{
        background-color: rgba(238, 227, 166, 0.6) !important;
        border: 1px solid var(--red) !important;
        transform: scale(1.05) !important;
    }}

    /* Floating Circular Collapse Button (in Expanded Mode) */
    [data-testid="stSidebarCollapseButton"] {{
        position: absolute !important;
        right: -14px !important;
        top: 26px !important;
        z-index: 999999 !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        border-radius: 50% !important;
        background: var(--red) !important;
        color: #ffffff !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 6px rgba(41, 41, 37, 0.25) !important;
        padding: 0 !important;
        cursor: pointer !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{
        background: #782228 !important;
        transform: scale(1.1) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button span {{
        color: #ffffff !important;
        font-size: 16px !important;
    }}

    /* Floating Circular Expand Button (in Collapsed Rail Mode) */
    [data-testid="stSidebarCollapsedControl"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        left: 54px !important;
        top: 26px !important;
        z-index: 999999 !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }}
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"] {{
        position: fixed !important;
        left: 54px !important;
        top: 26px !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        border-radius: 50% !important;
        background: var(--red) !important;
        color: #ffffff !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 6px rgba(41, 41, 37, 0.25) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 999999 !important;
        cursor: pointer !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }}
    [data-testid="stExpandSidebarButton"]:hover,
    button[data-testid="stExpandSidebarButton"]:hover {{
        background: #782228 !important;
        transform: scale(1.1) !important;
    }}
    [data-testid="stExpandSidebarButton"] span,
    button[data-testid="stExpandSidebarButton"] span {{
        color: #ffffff !important;
        font-size: 16px !important;
    }}

    /* =========================================================================
       AUTO-STABILIZATION & FLUID RESPONSIVE LAYOUT (NO WASTED LEFT/RIGHT SPACE)
       ========================================================================= */

    .main,
    [data-testid="stMain"] {{
        flex: 1 1 auto !important;
        min-width: 0 !important;
        transition: width 0.22s cubic-bezier(0.16, 1, 0.3, 1), margin-left 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    /* Dynamic fluid main container widths avoiding wasted margin gaps */
    [data-testid="stSidebar"][data-collapsed="true"] + div,
    [data-testid="stSidebar"][data-collapsed="true"] ~ .main,
    [data-testid="stSidebar"][data-collapsed="true"] ~ [data-testid="stMain"] {{
        width: calc(100% - 68px) !important;
        margin-left: 0 !important;
    }}

    [data-testid="stSidebar"][data-collapsed="false"] + div,
    [data-testid="stSidebar"][data-collapsed="false"] ~ .main,
    [data-testid="stSidebar"][data-collapsed="false"] ~ [data-testid="stMain"] {{
        width: calc(100% - 260px) !important;
        margin-left: 0 !important;
    }}

    /* Auto-stabilize block container padding across laptop/desktop screens */
    .main .block-container,
    [data-testid="stAppViewBlockContainer"] {{
        max-width: 100% !important;
        padding-top: 1.75rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: clamp(1.2rem, 2.5vw, 2.8rem) !important;
        padding-right: clamp(1.2rem, 2.5vw, 2.8rem) !important;
        box-sizing: border-box !important;
    }}

    /* Responsive column stabilization preventing squished cards and wrapped badges */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.85rem !important;
        align-items: stretch !important;
    }}

    .cr-card, .plan-card, .metric-card {{
        box-sizing: border-box !important;
        max-width: 100% !important;
    }}

    /* Flat Paper Surface (Dot texture removed per Stage F1) */
    body:before,
    .stApp:before,
    [data-testid="stAppViewContainer"]:before {{
        display: none !important;
        content: none !important;
    }}

    /* Typography */
    h1, h2, h3, .editorial-heading,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    [data-testid="stSubheader"] {{
        font-family: var(--display) !important;
        font-weight: 500 !important;
        letter-spacing: -0.055em !important;
        line-height: 1.13 !important;
        color: var(--ink) !important;
    }}

    h1, [data-testid="stHeading"] h1 {{
        font-size: clamp(2.4rem, 4.5vw, 4.2rem) !important;
        margin-bottom: 0.25rem !important;
    }}

    h2, [data-testid="stHeading"] h2 {{
        font-size: clamp(1.8rem, 3.2vw, 2.8rem) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }}

    h3, [data-testid="stHeading"] h3, [data-testid="stSubheader"] {{
        font-size: clamp(1.2rem, 2.0vw, 1.6rem) !important;
    }}

    em {{
        font-style: normal !important;
        color: var(--red) !important;
    }}

    /* Sidebar Navigation System */



    .cr-sidebar-brand {{
        padding: 2px 0 8px;
        border-bottom: 1.5px solid var(--line);
        margin-bottom: 8px;
    }}

    .cr-brand-title {{
        font-family: var(--display);
        font-size: 1.45rem;
        font-weight: 500;
        letter-spacing: -0.05em;
        line-height: 1.1;
        color: var(--ink);
        margin: 0;
    }}

    .cr-brand-title span {{
        color: var(--red);
    }}

    .cr-brand-subtitle {{
        font-family: var(--body);
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }}

    .cr-nav-group {{
        margin-bottom: 6px;
    }}

    .cr-nav-group-title {{
        font-family: var(--body);
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        padding: 0 4px 2px;
        border-bottom: 1px solid rgba(182, 175, 163, 0.4);
        margin-bottom: 2px;
    }}

    /* High Specificity: Strip default blue link colors and underlines from sidebar */
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a:link,
    [data-testid="stSidebar"] a:visited,
    [data-testid="stSidebar"] .stMarkdown a,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a,
    [data-testid="stSidebar"] a.cr-nav-link,
    a.cr-nav-link,
    .cr-nav-link {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 5px 10px !important;
        margin: 1px 0 !important;
        text-decoration: none !important;
        color: var(--ink) !important;
        font-family: var(--display) !important;
        font-size: 0.88rem !important;
        line-height: 1.25 !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        border-radius: 4px !important;
        border: 1px solid transparent !important;
        border-bottom: 1px dashed var(--line) !important;
        background-color: transparent !important;
        transition: transform 0.2s ease, background-color 0.2s ease, color 0.2s ease, border-left-color 0.2s ease !important;
        cursor: pointer !important;
    }}

    [data-testid="stSidebar"] a:hover,
    [data-testid="stSidebar"] .stMarkdown a:hover,
    [data-testid="stSidebar"] a.cr-nav-link:hover,
    a.cr-nav-link:hover,
    .cr-nav-link:hover {{
        transform: translateX(6px) !important;
        background-color: rgba(238, 227, 166, 0.45) !important;
        color: var(--red) !important;
        text-decoration: none !important;
        border-bottom: 1px solid var(--red) !important;
        border-left: 4px solid var(--red) !important;
    }}

    [data-testid="stSidebar"] a.cr-nav-link.is-active,
    [data-testid="stSidebar"] a.cr-nav-link[aria-current="page"],
    [data-testid="stSidebar"] .stMarkdown a.is-active,
    a.cr-nav-link.is-active,
    .cr-nav-link.is-active,
    .cr-nav-link[aria-current="page"] {{
        background-color: var(--yellow) !important;
        border-left: 5px solid var(--red) !important;
        border-bottom: 1px solid var(--line) !important;
        border-top: 1px solid rgba(182, 175, 163, 0.4) !important;
        border-right: 1px solid rgba(182, 175, 163, 0.4) !important;
        box-shadow: 2px 3px 6px rgba(50, 43, 28, 0.14) !important;
        transform: rotate(-0.8deg) scale(1.02) !important;
        color: var(--red) !important;
        font-weight: 700 !important;
        text-decoration: none !important;
    }}

    [data-testid="stSidebar"] a svg,
    [data-testid="stSidebar"] .cr-nav-link svg,
    .cr-nav-link svg {{
        flex-shrink: 0 !important;
        width: 20px !important;
        height: 20px !important;
        stroke: currentColor !important;
        stroke-width: 1.8 !important;
        fill: none !important;
        color: var(--muted) !important;
        transition: stroke 0.2s ease, color 0.2s ease, transform 0.2s ease !important;
    }}

    [data-testid="stSidebar"] a:hover svg,
    [data-testid="stSidebar"] a.cr-nav-link:hover svg,
    .cr-nav-link:hover svg {{
        color: var(--red) !important;
        stroke: var(--red) !important;
        transform: scale(1.1) !important;
    }}

    [data-testid="stSidebar"] a.cr-nav-link.is-active svg,
    .cr-nav-link.is-active svg {{
        color: var(--red) !important;
        stroke: var(--red) !important;
        transform: scale(1.05) !important;
    }}

    .cr-nav-link:focus-visible {{
        outline: 3px solid var(--blue) !important;
        outline-offset: 2px !important;
    }}

    /* Section Banner System with Aligned Local SVG Icon Badge */
    .cr-section-banner {{
        padding: 6px 0 20px;
        margin-bottom: 22px;
        border-bottom: 1.5px solid var(--line);
        position: relative;
    }}

    .cr-section-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }}

    .cr-section-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: rgba(152, 43, 50, 0.08);
        border: 1.5px solid var(--red);
        color: var(--red);
        flex-shrink: 0;
        box-shadow: 1px 2px 4px rgba(50, 43, 28, 0.1);
    }}

    .cr-section-icon svg {{
        width: 18px;
        height: 18px;
        stroke: var(--red);
        fill: none;
        stroke-width: 1.8;
    }}

    .cr-eyebrow-text {{
        font-family: var(--body);
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--red);
    }}

    .cr-section-title {{
        margin: 4px 0 6px !important;
        line-height: 1.15 !important;
    }}

    .cr-section-subtitle {{
        font-family: var(--body);
        font-size: 0.96rem;
        color: var(--muted);
        max-width: 800px;
        margin: 0;
        line-height: 1.5;
    }}

    /* Tactile Buttons */
    .button,
    button[kind="primary"],
    .stButton > button {{
        display: inline-flex !important;
        align-items: center !important;
        gap: 15px !important;
        text-decoration: none !important;
        border: 2px solid var(--red) !important;
        padding: 10px 22px !important;
        font-family: var(--display) !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        background-color: var(--paper) !important;
        color: var(--red) !important;
        border-radius: 0 !important;
        box-shadow: 4px 4px 0 var(--red) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        cursor: pointer !important;
    }}

    .button:hover,
    button[kind="primary"]:hover,
    .stButton > button:hover {{
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0 var(--red) !important;
        background-color: #fff9ee !important;
        color: var(--red) !important;
    }}

    .button:active,
    button[kind="primary"]:active,
    .stButton > button:active {{
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0 var(--red) !important;
    }}

    .stDownloadButton > button {{
        display: inline-flex !important;
        align-items: center !important;
        text-decoration: none !important;
        border: 2px solid var(--blue) !important;
        padding: 10px 22px !important;
        font-family: var(--display) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        background-color: var(--paper) !important;
        color: var(--blue) !important;
        border-radius: 0 !important;
        box-shadow: 4px 4px 0 var(--blue) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}

    .stDownloadButton > button:hover {{
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0 var(--blue) !important;
        background-color: #f0f4f8 !important;
        color: var(--blue) !important;
    }}

    .text-link, .under-link {{
        font-family: var(--display);
        text-decoration: none;
        border-bottom: 1px solid var(--line);
        padding-bottom: 3px;
        color: var(--ink);
        transition: color 0.2s ease, border-color 0.2s ease;
    }}

    .text-link:hover, .under-link:hover {{
        color: var(--red);
        border-color: var(--red);
    }}

    /* Tabs */
    [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 2px solid var(--line) !important;
        gap: 10px !important;
    }}

    [data-baseweb="tab"] {{
        font-family: var(--display) !important;
        font-size: 1.05rem !important;
        color: var(--muted) !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease !important;
    }}

    [data-baseweb="tab"]:hover {{
        color: var(--red) !important;
        background-color: rgba(238, 227, 166, 0.25) !important;
    }}

    [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--red) !important;
        font-weight: 700 !important;
        border-bottom: 3px solid var(--red) !important;
    }}


    /* Screen Auto-Stabilization: eliminate layout shifts, stale dimming, and running flicker */
    [data-stale="true"],
    .stApp [data-stale="true"],
    .stApp [data-stale="true"] *,
    [data-testid="stAppViewBlockContainer"] [data-stale="true"],
    [data-test-script-state="running"] [data-testid="stMain"],
    [data-test-script-state="running"] .main,
    [data-test-script-state="running"] [data-testid="stSidebar"],
    .stApp[data-test-script-state="running"] [data-testid="stAppViewBlockContainer"] {{
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }}

    /* Stabilize tab transitions and eliminate fadeIn blinking */
    [data-testid="stTabs"] {{
        min-height: 480px;
    }}
    [data-testid="stTabContent"] {{
        animation: none !important;
        transition: none !important;
    }}

    /* Innovative Pill styling */
    [data-testid="stPills"] {{
        margin: 6px 0 10px !important;
    }}
    [data-testid="stPills"] button,
    button[data-testid="stBaseButton-pills"] {{
        border-radius: 20px !important;
        border: 1.5px solid var(--line) !important;
        background-color: #ffffff !important;
        color: var(--ink) !important;
        font-family: var(--body) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 4px 14px !important;
        box-shadow: 1px 1px 0 rgba(41, 41, 37, 0.08) !important;
        transition: all 0.12s ease-in-out !important;
    }}
    [data-testid="stPills"] button:hover,
    button[data-testid="stBaseButton-pills"]:hover {{
        border-color: var(--red) !important;
        color: var(--red) !important;
        background-color: #fffaf7 !important;
        transform: translateY(-1px) !important;
        box-shadow: 1px 2px 4px rgba(152, 43, 50, 0.15) !important;
    }}
    [data-testid="stPills"] button[aria-pressed="true"],
    button[data-testid="stBaseButton-pills"][aria-pressed="true"] {{
        background-color: var(--red) !important;
        color: #ffffff !important;
        border-color: var(--red) !important;
        box-shadow: 2px 2px 0 #6e1c22 !important;
    }}

    /* Inline code and code blocks: warm parchment palette (never dark-mode black boxes) */
    code, pre,
    [data-testid="stMarkdownContainer"] code,
    [data-testid="stCaptionContainer"] code {{
        background-color: #eee9dd !important;
        color: #982b32 !important;
        border: 1px solid #d5cebf !important;
        border-radius: 3px !important;
        padding: 1px 6px !important;
        font-family: "Consolas", "Courier New", monospace !important;
        font-size: 0.84em !important;
    }}

    /* Form Controls: strict light paper enforcement across inputs, date pickers, number steppers, popovers */
    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"],
    [data-testid="stDateInput"] > div > div,
    [data-testid="stNumberInput"] > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: #fffdfa !important;
        border-color: var(--line) !important;
        border-radius: 0 !important;
        color: var(--ink) !important;
        font-family: var(--body) !important;
    }}
    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] > div > div,
    [data-testid="stNumberInput"] > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {{
        border: 1.5px solid var(--line) !important;
        box-shadow: 2px 2px 0 rgba(41, 41, 37, 0.08) !important;
        background-color: #fffdfa !important;
    }}
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {{
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        background-color: #fffdfa !important;
    }}
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {{
        background-color: #eee9dd !important;
        color: var(--ink) !important;
        border-left: 1px solid var(--line) !important;
        box-shadow: none !important;
    }}
    [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stNumberInputStepUp"]:hover {{
        background-color: var(--yellow) !important;
        color: var(--red) !important;
    }}

    /* Dropdown menus, popovers, calendars */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="calendar"],
    ul[role="listbox"] {{
        background-color: #fffdfa !important;
        color: var(--ink) !important;
    }}
    ul[role="listbox"] li,
    [data-baseweb="menu"] li {{
        color: var(--ink) !important;
        background-color: #fffdfa !important;
    }}
    ul[role="listbox"] li:hover,
    [data-baseweb="menu"] li:hover {{
        background-color: #f0ebe0 !important;
        color: var(--red) !important;
    }}

    [data-baseweb="select"] > div:focus-within,
    [data-testid="stDateInput"] > div > div:focus-within,
    [data-testid="stNumberInput"] > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--red) !important;
        box-shadow: 3px 3px 0 var(--red) !important;
        outline: none !important;
    }}

    /* All widget, radio, checkbox, and caption labels (prevent OS dark mode white text on light paper) */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-baseweb="radio"] div,
    [data-baseweb="radio"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label *,
    [data-baseweb="checkbox"] div,
    [data-baseweb="checkbox"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: var(--ink) !important;
    }}
    label[data-testid="stWidgetLabel"] p {{
        font-family: var(--display) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: #fffdfa !important;
        border: 1.5px solid var(--line) !important;
        border-radius: 0 !important;
        box-shadow: 3px 3px 0 rgba(41, 41, 37, 0.06) !important;
        margin-bottom: 16px !important;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *:not([data-testid="stIconMaterial"]) {{
        font-family: var(--display) !important;
        font-size: 1.05rem !important;
        color: var(--ink) !important;
    }}

    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded" !important;
        font-size: 1.25rem !important;
    }}

    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover *:not([data-testid="stIconMaterial"]) {{
        color: var(--red) !important;
    }}

    /* Dataframe and Tables */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        color-scheme: light !important;
        border: 1.5px solid var(--line) !important;
        box-shadow: 4px 4px 0 rgba(41, 41, 37, 0.08) !important;
        background-color: #ffffff !important;
        color: var(--ink) !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        font-family: var(--display) !important;
        font-size: 2.2rem !important;
        color: var(--ink) !important;
        letter-spacing: -0.05em !important;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p {{
        font-family: var(--body) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 0.74rem !important;
        color: var(--muted) !important;
        font-weight: 700 !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 0 !important;
        border-left: 4px solid var(--red) !important;
        background-color: var(--panel) !important;
        color: var(--ink) !important;
        box-shadow: 2px 2px 5px rgba(50, 43, 28, 0.08) !important;
    }}
    [data-testid="stAlert"] p {{
        color: var(--ink) !important;
    }}

    /* Hero Layout */
    .hero-container {{
        display: grid;
        grid-template-columns: 1.4fr 0.6fr;
        gap: 45px;
        align-items: start;
        padding: 25px 0 55px;
        position: relative;
        border-bottom: 1px solid var(--line);
        margin-bottom: 30px;
    }}

    .eyebrow {{
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.74rem;
        font-weight: 700;
        color: var(--red);
        margin-bottom: 16px;
        font-family: var(--body);
    }}

    .hero-container h1 {{
        font-size: clamp(3.0rem, 5.2vw, 5.2rem);
        letter-spacing: -0.075em;
        line-height: 1.08;
        margin: 0;
    }}

    .hero-underline {{
        width: 65%;
        height: 34px;
        display: block;
        margin-left: 18%;
        margin-top: -4px;
        fill: none;
        stroke: var(--red);
        stroke-width: 2.2;
    }}

    .hero-actions {{
        display: flex;
        align-items: center;
        gap: 25px;
        margin-top: 34px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}

    .hero-caption {{
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px dashed rgba(182, 175, 163, 0.45);
        max-width: 680px;
        font-family: var(--body);
    }}

    .hero-right {{
        position: relative;
        padding-top: 15px;
    }}

    .hero-intro {{
        font-family: var(--display);
        font-size: 1.15rem;
        line-height: 1.6;
        color: var(--ink);
        margin-bottom: 20px;
    }}

    .doodle-star {{
        width: 65px;
        height: 65px;
        position: absolute;
        top: -15px;
        left: -75px;
        fill: none;
        stroke: var(--red);
        stroke-width: 1.8;
    }}

    .hero-photo {{
        width: min(100%, 340px);
        margin: 20px 0 0;
        padding: 8px 8px 11px;
        background: #fffdf8;
        box-shadow: 0 8px 19px rgba(53, 45, 33, 0.15);
        transform: rotate(2deg);
        overflow: hidden;
        position: relative;
        transition: transform .3s ease, box-shadow .3s ease;
        border: 1px solid var(--line);
    }}

    .hero-photo img {{
        display: block;
        width: 100%;
        height: 180px;
        object-fit: cover;
        transition: transform .35s ease;
    }}

    .hero-photo figcaption {{
        padding: 8px 4px 0;
        font-family: var(--display);
        font-size: 0.78rem;
        text-align: center;
        color: var(--blue);
    }}

    .hero-photo:hover {{
        z-index: 3;
        transform: rotate(0deg) scale(1.065);
        box-shadow: 0 0 0 3px #fff9ee, 0 0 22px 5px rgba(228, 182, 83, 0.6), 0 14px 30px rgba(53, 45, 33, 0.25);
    }}

    .hero-photo:hover img {{
        transform: scale(1.06);
    }}

    .sticky {{
        background: var(--yellow);
        padding: 24px 28px;
        width: max-content;
        max-width: 100%;
        font-family: var(--display);
        font-size: 1rem;
        text-align: center;
        transform: rotate(-3deg);
        position: relative;
        margin: 22px 0 0 20px;
        box-shadow: 3px 6px 7px rgba(50, 43, 28, 0.12);
        border: 1px solid #dfd6a8;
    }}

    .tape {{
        position: absolute;
        width: 70px;
        height: 20px;
        background: rgba(237, 230, 194, 0.75);
        top: -10px;
        left: 50%;
        transform: translateX(-50%) rotate(3deg);
    }}

    .hero-arrow {{
        width: 130px;
        position: absolute;
        bottom: -15px;
        left: 55%;
        fill: none;
        stroke: var(--red);
        stroke-width: 1.8;
        stroke-linecap: round;
    }}

    /* Scoped Stats Strip */
    .stats {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        padding: 30px 0 40px;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        margin: 25px 0 35px;
        gap: 20px;
    }}

    .stat {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}

    .stat strong {{
        font-family: var(--display);
        font-size: 2.6rem;
        font-weight: 500;
        letter-spacing: -0.07em;
        line-height: 1;
        color: var(--ink);
    }}

    .stat span {{
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.35;
        font-family: var(--body);
    }}

    /* Feature Grid */
    .feature-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 35px 60px;
        margin: 30px 0 45px;
    }}

    .feature {{
        display: flex;
        align-items: flex-start;
        gap: 22px;
    }}

    .feature-icon {{
        display: grid;
        place-items: center;
        width: 48px;
        height: 48px;
        flex: none;
        border: 1.5px solid var(--red);
        border-radius: 50%;
        font-family: var(--display);
        font-size: 1.3rem;
        color: var(--red);
        transform: rotate(-12deg);
    }}

    .feature:nth-child(2) .feature-icon,
    .feature:nth-child(3) .feature-icon {{
        color: var(--blue);
        border-color: var(--blue);
    }}

    .feature h3 {{
        font-family: var(--display);
        font-size: 1.25rem;
        margin: 0 0 8px;
    }}

    .feature p {{
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.5;
        margin: 0;
        font-family: var(--body);
    }}

    /* Category Cards */
    .place-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-top: 30px;
        margin-bottom: 45px;
    }}

    .place-card {{
        min-width: 0;
        position: relative;
        transition: transform .3s ease;
    }}

    .place-card.lifted {{
        transform: translateY(20px);
    }}

    .place-art {{
        height: 190px;
        border: 8px solid #fff;
        box-shadow: 0 4px 16px rgba(59, 48, 37, 0.12);
        display: block;
        overflow: hidden;
        position: relative;
        margin-bottom: 16px;
        transition: transform .3s ease, box-shadow .3s ease;
    }}

    .place-art img {{
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
        transition: transform .35s ease;
    }}

    .card-kicker {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--red);
        font-weight: 700;
        display: block;
        font-family: var(--body);
    }}

    .place-card h3 {{
        font-family: var(--display);
        font-size: 1.22rem;
        letter-spacing: -0.04em;
        margin: 6px 0;
    }}

    .place-card p {{
        font-size: 0.85rem;
        color: var(--muted);
        line-height: 1.45;
        font-family: var(--body);
    }}

    .place-card:hover {{
        z-index: 5;
    }}

    .place-card:hover .place-art {{
        transform: translateY(-5px) scale(1.075);
        box-shadow: 0 0 0 3px #fff9ee, 0 0 22px 5px rgba(228, 182, 83, 0.65), 0 13px 25px rgba(53, 45, 33, 0.2);
    }}

    .place-card:hover .place-art img {{
        transform: scale(1.075);
    }}

    /* Workflow Panel */
    .workflow-panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 38px 40px 44px;
        transform: rotate(-0.4deg);
        margin: 40px 0 50px;
    }}

    .workflow-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 30px;
        border-bottom: 1px solid var(--line);
        padding-bottom: 18px;
    }}

    .workflow-head h2 {{
        font-family: var(--display);
        font-size: clamp(1.8rem, 2.8vw, 2.6rem);
        margin: 0;
    }}

    .workflow-head span {{
        color: var(--muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: var(--body);
    }}

    .steps {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        margin-top: 30px;
    }}

    .step {{
        position: relative;
        padding-right: 20px;
    }}

    .step:not(:last-child):after {{
        content: "→";
        position: absolute;
        right: -16px;
        top: 10px;
        font-family: var(--display);
        font-size: 2rem;
        color: var(--red);
    }}

    .step b {{
        font-family: var(--display);
        font-weight: 500;
        color: var(--red);
        display: inline-grid;
        place-items: center;
        border: 1.5px solid var(--red);
        width: 32px;
        height: 32px;
        transform: rotate(-7deg);
        margin-bottom: 12px;
    }}

    .step h3 {{
        font-family: var(--display);
        font-size: 1.15rem;
        margin: 0 0 8px;
    }}

    .step p {{
        font-size: 0.84rem;
        color: var(--muted);
        line-height: 1.45;
        margin: 0;
        font-family: var(--body);
    }}

    /* Plan Cards */
    .plan-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin: 25px 0 35px;
    }}

    .plan-card {{
        position: relative;
        border: 1.5px solid var(--line);
        padding: 26px 30px 28px;
        background: rgba(255, 255, 255, 0.7);
        min-height: 270px;
        box-shadow: 4px 4px 0 rgba(41, 41, 37, 0.06);
    }}

    .featured-plan {{
        background: #f1e8b6 !important;
        border-color: var(--red) !important;
        transform: rotate(-0.7deg);
        box-shadow: 4px 6px 12px rgba(50, 43, 28, 0.12);
    }}

    .plan-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding-bottom: 12px;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .plan-top strong {{
        font-size: 0.75rem;
        color: var(--red);
        font-weight: 700;
    }}

    .plan-number {{
        font-family: var(--display);
        font-size: 3.2rem;
        line-height: 1;
        margin-top: 16px;
        letter-spacing: -0.08em;
    }}

    .plan-card > p {{
        color: var(--muted);
        font-size: 0.8rem;
        margin: 4px 0 15px;
    }}

    .plan-details {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px 18px;
        margin-top: 15px;
    }}

    .plan-details span {{
        font-size: 0.8rem;
        line-height: 1.4;
    }}

    .plan-details b {{
        color: var(--red);
    }}

    .plan-tag {{
        position: absolute;
        top: -15px;
        right: 25px;
        background: var(--red);
        color: white;
        padding: 4px 12px;
        font-family: var(--display);
        font-size: 0.72rem;
        transform: rotate(4deg);
        font-weight: 600;
        box-shadow: 2px 2px 4px rgba(41, 41, 37, 0.2);
    }}

    /* Badges without emojis */
    .tag-synthetic {{
        display: inline-block;
        background: #ede8dc;
        color: var(--ink);
        border: 1px solid #c8c0b2;
        padding: 2px 8px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: var(--body);
    }}

    .tag-status {{
        display: inline-block;
        background: #e9eff5;
        color: var(--quiet-blue);
        border: 1px solid #b8cbdc;
        padding: 2px 8px;
        font-size: 0.74rem;
        font-weight: 600;
        font-family: var(--body);
    }}

    /* Focus visible */
    a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible {{
        outline: 3px solid var(--blue) !important;
        outline-offset: 3px !important;
    }}

    /* Responsive */
    @media (max-width: 900px) {{
        .hero-container {{
            grid-template-columns: 1fr;
            gap: 25px;
        }}
        .stats {{
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        .place-grid {{
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .feature-grid {{
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        .steps {{
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        .plan-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    @media (max-width: 540px) {{
        .stats {{
            grid-template-columns: 1fr;
        }}
        .place-grid {{
            grid-template-columns: 1fr;
        }}
        .place-card.lifted {{
            transform: none;
        }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *:before, *:after {{
            transition: none !important;
            animation: none !important;
            transform: none !important;
        }}
    }}

    {reduced_motion_css}
</style>
    """
    return css


def render_sidebar_navigation(active_view: str) -> None:
    """Render real link navigation with aligned local SVG icons and page-level URL routing."""
    active_norm = active_view.strip().lower()

    # SVG Icon Definitions (stroke 1.8, 18x18, aria-hidden)
    icons = {
        "overview": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "discover": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
        "planner": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="19" r="3"/><circle cx="18" cy="5" r="3"/><path d="M12 19h4.5a3.5 3.5 0 0 0 0-7h-9a3.5 3.5 0 0 1 0-7H12"/></svg>',
        "compare": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "datasets": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "insights": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "devlab": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 2v7.31L4.1 20.1A2 2 0 0 0 5.8 23h12.4a2 2 0 0 0 1.7-2.9L14 9.31V2"/><line x1="8.5" y1="2" x2="15.5" y2="2"/><line x1="6.8" y1="15" x2="17.2" y2="15"/></svg>',
        "model": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "settings": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "information": '<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    }

    groups = [
        {
            "name": "EXPLORE",
            "items": [
                ("overview", "Overview"),
                ("discover", "Discover"),
                ("planner", "Planner"),
                ("compare", "Compare"),
            ],
        },
        {
            "name": "STUDY",
            "items": [
                ("datasets", "Datasets"),
                ("insights", "Insights"),
                ("devlab", "DEV Lab"),
                ("model", "Prediction Model"),
            ],
        },
        {
            "name": "HELP",
            "items": [
                ("settings", "Settings"),
                ("information", "Information"),
            ],
        },
    ]

    html = """
    <div class="cr-sidebar-brand">
        <div class="cr-brand-title" title="CultureRoute"><span class="cr-brand-full">Culture<span>Route</span></span><span class="cr-brand-mini">CR</span></div>
        <div class="cr-brand-subtitle">DEV AAT · Cultural Tourism Study</div>
    </div>
    <nav class="cr-sidebar-nav" aria-label="Main Navigation">
    """

    for g in groups:
        html += f'<div class="cr-nav-group"><div class="cr-nav-group-title">{g["name"]}</div>'
        for key, label in g["items"]:
            is_active = (active_norm == key) or (active_norm == label.lower())
            if key == "devlab" and active_norm in ["lab", "study", "devlab"]:
                is_active = True
            elif key == "model" and active_norm in ["model", "prediction", "prediction model"]:
                is_active = True
            elif key == "information" and active_norm in ["information", "info"]:
                is_active = True

            active_attr = ' aria-current="page" class="cr-nav-link is-active"' if is_active else ' class="cr-nav-link"'
            icon_svg = icons.get(key, "")
            html += f'<a href="?view={key}" target="_self"{active_attr} title="{label}">{icon_svg}<span>{label}</span></a>'
        html += "</div>"

    html += """
    </nav>
    <div class="sticky" style="background: #eee3a6; padding: 8px 10px; font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; font-size: 0.70rem; line-height: 1.3; color: #292925; transform: rotate(-1deg); position: relative; margin: 8px 2px 4px; box-shadow: 1px 2px 5px rgba(50, 43, 28, 0.10); border: 1px solid #dfd6a8;">
        <span class="tape" style="position: absolute; width: 36px; height: 10px; background: rgba(237, 230, 194, 0.8); top: -5px; left: 50%; transform: translateX(-50%) rotate(1.5deg);"></span>
        <div style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #982b32; margin-bottom: 2px;">Study Data Boundary</div>
        <p style="margin: 0; font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; font-size: 0.68rem; line-height: 1.25; color: #292925;">
            Read-only 45-city study; separate India extract catalog. Real routing is opt-in.
        </p>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)




SECTION_ICONS: dict[str, str] = {
    "overview": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/>'
        '</svg>'
    ),
    "discover": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
        '</svg>'
    ),
    "planner": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="6" cy="19" r="3"/>'
        '<circle cx="18" cy="5" r="3"/>'
        '<path d="M12 19h4.5a3.5 3.5 0 0 0 0-7h-9a3.5 3.5 0 0 1 0-7H12"/>'
        '</svg>'
    ),
    "compare": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
        '<path d="M2 20h20"/>'
        '</svg>'
    ),
    "datasets": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
        '</svg>'
    ),
    "insights": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/>'
        '</svg>'
    ),
    "devlab": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M10 2v7.31L4.1 20.1A2 2 0 0 0 5.8 23h12.4a2 2 0 0 0 1.7-2.9L14 9.31V2"/>'
        '<line x1="8.5" y1="2" x2="15.5" y2="2"/>'
        '<line x1="6.8" y1="15" x2="17.2" y2="15"/>'
        '</svg>'
    ),
    "model": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
        '</svg>'
    ),
    "settings": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<line x1="4" y1="21" x2="4" y2="14"/>'
        '<line x1="4" y1="10" x2="4" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12" y2="3"/>'
        '<line x1="20" y1="21" x2="20" y2="16"/>'
        '<line x1="20" y1="12" x2="20" y2="3"/>'
        '<line x1="1" y1="14" x2="7" y2="14"/>'
        '<line x1="9" y1="8" x2="15" y2="8"/>'
        '<line x1="17" y1="16" x2="23" y2="16"/>'
        '</svg>'
    ),
    "information": (
        '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
        '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'
        '<line x1="9" y1="7" x2="15" y2="7"/>'
        '<line x1="9" y1="11" x2="15" y2="11"/>'
        '</svg>'
    ),
}


def render_section_banner(
    section_key: str,
    title_html: str,
    subtitle: str = "",
    kicker: str = "",
) -> None:
    """Render an authoritative editorial section banner with aligned local SVG icon badge."""
    icon_svg = SECTION_ICONS.get(section_key.lower().strip(), "")
    kicker_part = f'<span class="cr-eyebrow-text">{kicker}</span>' if kicker else ""
    subtitle_part = f'<p class="cr-section-subtitle">{subtitle}</p>' if subtitle else ""

    html = f"""
    <div class="cr-section-banner">
        <div class="cr-section-eyebrow">
            <span class="cr-section-icon" aria-hidden="true">{icon_svg}</span>
            {kicker_part}
        </div>
        <h1 class="editorial-heading cr-section-title">{title_html}</h1>
        {subtitle_part}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_editorial_hero() -> None:
    """Render the editorial split hero with truthful demo actions, framed photo, and taped sticky note."""
    img_landscape = get_asset_base64("green-landscape.jpg")

    html = f"""
    <div class="hero-container">
        <div>
            <div class="eyebrow">Cultural tourism / India / DEV AAT</div>
            <h1 class="editorial-heading">Explore the culture<br>behind <em>every route.</em></h1>
            <svg class="hero-underline" viewBox="0 0 460 34" aria-hidden="true">
                <path d="M4 22C115 6 294 8 455 16M72 31c111-14 256-13 361-7"/>
            </svg>
            <div class="hero-actions">
                <a class="button" href="?view=discover">Explore Study Cities <span aria-hidden="true">↗</span></a>
                <a class="text-link" href="?view=devlab">See the DEV Lab Study <span aria-hidden="true">→</span></a>
            </div>
            <p class="hero-caption">
                An offline study of cultural tourism representation across 45 Indian cities,
                examining the balance between heritage preservation and travel efficiency.
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="tag-synthetic">45 Indian Cities</span>
                <span class="tag-synthetic">675 Attractions</span>
                <span class="tag-synthetic">24 Canonical Tables</span>
                <span class="tag-synthetic">Real Routing: Opt-In</span>
            </div>
        </div>
        <div class="hero-right">
            <svg class="doodle-star" viewBox="0 0 100 100" aria-hidden="true">
                <path d="M50 4 59 34 88 15 72 43 99 52 69 59 83 88 55 70 45 99 40 68 8 82 29 55 1 43 34 40 20 11 46 31Z"/>
            </svg>
            <p class="hero-intro">
                Find heritage sites, museums, religious places, and food experiences.
                Then see how the route and the evidence come together.
            </p>
            <figure class="hero-photo">
                <img src="{img_landscape}" alt="Regional Indian landscape illustration" />
                <figcaption>A place to begin exploring</figcaption>
            </figure>
            <div class="sticky">
                45 cities to explore.<br>One route at a time.
                <span class="tape" aria-hidden="true"></span>
            </div>
        </div>
        <svg class="hero-arrow" viewBox="0 0 150 85" aria-hidden="true">
            <path d="M5 29c56-36 82 5 39 30-25 14-34-26 14-26 27 0 42 22 70 33m-17-17 17 17-23 4"/>
        </svg>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_scoped_stats(
    cities: int = 45,
    attractions: int = 675,
    trips: int = 411,
    rows: int = 48728,
    cultural_listings: str = "31.3%",
    cultural_actions: str = "31.3%",
    cultural_stops: str = "30.1%",
    weeks: int = 57,
) -> None:
    """Render the 8 audited scoped statistics in the prototype two-tier format."""
    html = f"""
    <section class="stats" aria-label="Project scope">
        <div class="stat">
            <strong>{cities}</strong>
            <span>Indian cities<br>in the study</span>
        </div>
        <div class="stat">
            <strong>{attractions}</strong>
            <span>listed attractions<br>in scope</span>
        </div>
        <div class="stat">
            <strong>{trips}</strong>
            <span>non-cancelled trips<br>in the analysis</span>
        </div>
        <div class="stat">
            <strong>{rows:,}</strong>
            <span>distinct source rows<br>across 24 tables</span>
        </div>
    </section>

    <section class="stats" style="margin-top: -30px; border-top: none;" aria-label="Cultural representation shares">
        <div class="stat">
            <strong>{cultural_listings}</strong>
            <span>Listings Cultural Share<br>211 of 675 catalog attractions</span>
        </div>
        <div class="stat">
            <strong>{cultural_actions}</strong>
            <span>Interaction Cultural Share<br>2,144 of 6,850 matched events</span>
        </div>
        <div class="stat">
            <strong>{cultural_stops}</strong>
            <span>Planned Cultural Share<br>961 of 3,191 selected stops</span>
        </div>
        <div class="stat">
            <strong>{weeks}</strong>
            <span>Complete Forecast Weeks<br>Last-12 MAE: 3.42 (last-week)</span>
        </div>
    </section>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_feature_grid() -> None:
    """Render the 2x2 architectural feature grid matching the prototype."""
    html = """
    <div style="margin-top: 30px;">
        <div class="eyebrow">01 / Discover and decide</div>
        <h2 class="editorial-heading" style="margin-top: 0;">Everything you need<br>to make a <em>day count</em></h2>
        <p style="color: var(--muted); max-width: 550px; margin-top: -5px; font-size: 0.95rem;">
            Find the places that interest you, check what a route can cover, and keep the method in view.
        </p>
    </div>
    <div class="feature-grid">
        <article class="feature">
            <span class="feature-icon" aria-hidden="true">01</span>
            <div>
                <h3>Directed Travel Matrix</h3>
                <p>
                    10 attractions per city feature recorded directed walking and cab legs in the synthetic travel matrix.
                    The remaining 5 attractions are preserved as discovery-only catalog entries. Missing legs are never guessed.
                </p>
            </div>
        </article>
        <article class="feature">
            <span class="feature-icon" aria-hidden="true">02</span>
            <div>
                <h3>Dual-Objective Day Planner</h3>
                <p>
                    Deterministic itinerary engine ranking feasible schedules for minimum elapsed time versus maximum
                    cultural stop representation. Reproduces the Jaipur Saturday specification baseline with exact currency and carbon accounting.
                </p>
            </div>
        </article>
        <article class="feature">
            <span class="feature-icon" aria-hidden="true">03</span>
            <div>
                <h3>Transparent Statistical Methods</h3>
                <p>
                    Classical Wilson score intervals, conjugate Beta(1,1) Bayesian posteriors, and maximum-likelihood lognormal
                    footfall estimation. Strictly rule-based and transparent without opaque AI/ML black boxes.
                </p>
            </div>
        </article>
        <article class="feature">
            <span class="feature-icon" aria-hidden="true">04</span>
            <div>
                <h3>Evaluator DEV Lab</h3>
                <p>
                    Full 24-table schema catalog profiling, guided pandas operations, SQLite authorizer-enforced read-only sandbox,
                    and formula-safe CSV/Parquet export engines covering all 5 syllabus units.
                </p>
            </div>
        </article>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_category_cards() -> None:
    """Render the four photographic category cards with lifted layout and golden-glow hover."""
    img_fort = get_asset_base64("heritage-fort.avif")
    img_museum = get_asset_base64("museum-interior.jpg")
    img_temple = get_asset_base64("golden-temple.jpg")
    img_food = get_asset_base64("local-food.jpg")

    html = f"""
    <div style="margin-top: 35px;">
        <div class="eyebrow">A little inspiration</div>
        <h2 class="editorial-heading" style="margin-top: 0;">Culture is in the <em>details</em></h2>
        <p style="color: var(--muted); max-width: 550px; margin-top: -5px; font-size: 0.95rem;">
            Categories the project studies across attraction listings, online interactions, and planned stops.
        </p>
    </div>
    <div class="place-grid">
        <article class="place-card">
            <div class="place-art">
                <img src="{img_fort}" alt="Historic stone fort rising above a city" loading="lazy" />
            </div>
            <span class="card-kicker">01 / Heritage &amp; Forts</span>
            <h3>Stories in stone</h3>
            <p>Explore historic places and see whether they make it into a feasible day plan. (72 catalog attractions, 34.1%)</p>
        </article>
        <article class="place-card lifted">
            <div class="place-art">
                <img src="{img_museum}" alt="Ornate heritage interior with decorative walls" loading="lazy" />
            </div>
            <span class="card-kicker">02 / Museums &amp; Galleries</span>
            <h3>Rooms to remember</h3>
            <p>Compare catalog presence with saves and selected itinerary stops. (53 catalog attractions, 25.1%)</p>
        </article>
        <article class="place-card">
            <div class="place-art">
                <img src="{img_temple}" alt="The Golden Temple and surrounding water in Amritsar" loading="lazy" />
            </div>
            <span class="card-kicker">03 / Religious Sites</span>
            <h3>Places of meaning</h3>
            <p>Explore living heritage and see how religious sites appear in planned stops. (48 catalog attractions, 22.7%)</p>
        </article>
        <article class="place-card">
            <div class="place-art">
                <img src="{img_food}" alt="Indian meal with curries, rice, bread, and snacks" loading="lazy" />
            </div>
            <span class="card-kicker">04 / Food Experiences</span>
            <h3>Local flavor, local context</h3>
            <p>Make room for food experiences in a schedule that respects time and budget. (38 catalog attractions, 18.0%)</p>
        </article>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_workflow_panel() -> None:
    """Render the 3-step workflow panel matching the prototype."""
    html = """
    <div class="workflow-panel">
        <div class="workflow-head">
            <h2 class="editorial-heading">How CultureRoute works</h2>
            <span>From a city to a route to its evidence</span>
        </div>
        <div class="steps">
            <div class="step">
                <b>1</b>
                <h3>Pick your place</h3>
                <p>Choose one of the 45 Indian study cities, visit date, start time, budget, and desired stop count.</p>
            </div>
            <div class="step">
                <b>2</b>
                <h3>Shape the day</h3>
                <p>Evaluate recorded directed matrix legs, attraction opening hours, wait times, and entry fees.</p>
            </div>
            <div class="step">
                <b>3</b>
                <h3>See the reasoning</h3>
                <p>Compare Plan A (fastest) against Plan B (most cultural stops). Trace numbers to source data.</p>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
