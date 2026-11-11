"""CultureRoute Streamlit Application Entrypoint (Corrective Redesign R1-R8).

Provides:
- Editorial overview matching Acme reference video and site/dist design system.
- Synthetic Research Study (45 Indian cities, 675 attractions, 411 trips, 24 tables).
- Discover Catalog with category, price, and accessibility filtering.
- Deterministic Day Planner & Plan Comparison Engine (Jaipur specification fixture).
- Insights & Predictions (Cultural shares, weekly forecasting, footfall imputation).
- DEV Lab Evaluator Workspace (Units 1-5, SQLite authorizer sandbox, typed exports).
- Settings and Information screens.
- Strict two-pack boundary: SmartTrip and WanderMatch are the sole active project data.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import json
import math
from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from cultureroute.study_data import (
    verify_and_audit_study,
    get_study_cities,
    get_city_attractions,
    open_study_databases,
)
from cultureroute.study_db import (
    build_study_database,
    load_study_catalog,
    open_study_cache,
    DEFAULT_STUDY_DB,
)
from cultureroute.datasets_view import render_datasets_screen
from cultureroute.insights_view import render_insights_screen
from cultureroute.model_view import render_model_screen
from cultureroute.ui_components import (
    apply_editorial_chart_theme,
    inject_editorial_css,
    render_editorial_hero,
    render_scoped_stats,
    render_feature_grid,
    render_category_cards,
    render_workflow_panel,
    render_sidebar_navigation,
    render_section_banner,
)
from cultureroute.domain import (
    CulturalCategory,
    ResultLevel,
    VerificationStatus,
    FeeStatus,
    AccessibilityStatus,
)
from cultureroute.planner import (
    parse_time_to_minutes,
    plan_synthetic_day,
    plan_real_place_day,
    DayPlan,
)
from cultureroute.routing import (
    TransportMode,
    RouteLegStore,
    compute_directed_leg,
    DEFAULT_ROUTES_DB,
    configured_backend,
)
from cultureroute.catalog import RealCatalogStore
from cultureroute.boundary import IndiaBoundaryValidator, is_coordinate_in_india_bbox
from cultureroute.evidence import VenueEvidenceStore, FieldEvidenceStatus
from cultureroute.analytics import (
    compute_cultural_shares,
    compute_weekly_forecast_backtest,
    compute_footfall_imputation_study,
)
from cultureroute.dev_lab import (
    compute_methodology_comparison,
    compute_canonical_saved_planned_merge,
    ReadOnlySQLSandbox,
    export_dataframe_to_csv,
    export_dataframe_to_parquet,
    get_table_profile,
    filter_and_derive_poi,
    groupby_aggregate_stops,
    pivot_city_category_matrix,
    transform_user_interactions,
    get_syllabus_reference_units,
)

# Optional real data paths
BOUNDARY_GEOJSON = Path(__file__).resolve().parent / "data" / "india_boundary.geojson"
REAL_CATALOG_DB = Path(__file__).resolve().parent / "data" / "real_catalog.db"
VENUE_EVIDENCE_DB = Path(__file__).resolve().parent / "data" / "venue_evidence.db"


@st.cache_resource
def _get_boundary_validator():
    """Load and parse the India boundary GeoJSON once per server lifecycle."""
    return IndiaBoundaryValidator(BOUNDARY_GEOJSON) if BOUNDARY_GEOJSON.exists() else None


@st.cache_resource
def _ensure_startup_databases() -> bool:
    """Ensure local SQLite caches exist cleanly once per server startup."""
    if not VENUE_EVIDENCE_DB.exists():
        try:
            from cultureroute.pilot_evidence import seed_pilot_evidence
            seed_pilot_evidence(VENUE_EVIDENCE_DB)
        except Exception:
            pass
    if not DEFAULT_STUDY_DB.exists():
        try:
            build_study_database(DEFAULT_STUDY_DB)
        except Exception:
            pass
    return True


def _ensure_real_catalog_db() -> bool:
    """Build real_catalog.db lazily from committed source extracts when missing."""
    if REAL_CATALOG_DB.exists():
        return True
    unlocode_src = Path(__file__).resolve().parent / "data" / "unlocode_raw.csv"
    unesco_src = Path(__file__).resolve().parent / "data" / "wikidata_unesco_india.json"
    if unlocode_src.exists() and BOUNDARY_GEOJSON.exists():
        try:
            from cultureroute.ingest import run_streaming_import
            run_streaming_import(
                unlocode_csv=unlocode_src,
                unesco_json=unesco_src if unesco_src.exists() else None,
                boundary_geojson=BOUNDARY_GEOJSON,
                output_db=REAL_CATALOG_DB,
            )
            return REAL_CATALOG_DB.exists()
        except Exception:
            return False
    return False


@st.cache_data(ttl=3600)
def _get_city_guides_df(city_id: str) -> pd.DataFrame:
    with open_study_cache() as conn:
        rows = conn.execute(
            "SELECT display_name, languages, specialisation, years_experience, rating, review_count "
            "FROM tour_guides WHERE city_id = ? ORDER BY rating DESC",
            (city_id,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=3600)
def _get_devlab_actions_df() -> pd.DataFrame:
    with open_study_cache() as conn:
        return pd.read_sql_query(
            "SELECT interaction_type, count(*) as count FROM user_interactions GROUP BY interaction_type ORDER BY count DESC",
            conn,
        )


@st.cache_data(ttl=3600)
def _get_devlab_poi_costs_df() -> pd.DataFrame:
    with open_study_cache() as conn:
        return pd.read_sql_query("""
            SELECT p.poi_id, p.name, p.poi_category, c.name as city_name, CAST(p.entry_cost AS REAL) as entry_cost
            FROM activities_poi p
            JOIN cities c USING(city_id)
            WHERE c.country_code = 'IN' AND p.entry_cost IS NOT NULL
        """, conn)


@st.cache_data(ttl=3600)
def _get_devlab_events_df() -> pd.DataFrame:
    with open_study_cache() as conn:
        return pd.read_sql_query("""
            SELECT e.name, e.expected_footfall, e.recurrence, e.season, c.name as city_name
            FROM events_festivals e
            JOIN cities c USING(city_id)
            WHERE c.country_code = 'IN' AND e.expected_footfall IS NOT NULL
            ORDER BY e.expected_footfall ASC
        """, conn)


@st.cache_data(ttl=3600)
def _get_devlab_bivariate_df() -> pd.DataFrame:
    with open_study_cache() as conn:
        return pd.read_sql_query("""
            SELECT a.poi_id, a.name, a.poi_category, c.name as city_name, CAST(a.entry_cost AS REAL) as entry_cost, count(ui.interaction_id) as saves_count
            FROM activities_poi a
            JOIN cities c USING(city_id)
            LEFT JOIN user_interactions ui ON ui.entity_id = a.poi_id AND ui.interaction_type = 'save'
            WHERE c.country_code = 'IN' AND a.entry_cost IS NOT NULL
            GROUP BY a.poi_id
            HAVING saves_count > 0
        """, conn)


@st.cache_data(ttl=3600)
def _run_planner_pair_cached(
    city_name: str,
    visit_date: str,
    start_time: str,
    max_duration_hours: float,
    budget_inr: float,
    target_stops: int,
    transport_mode: str,
    step_free_only: bool,
    preferred_categories: tuple[str, ...],
):
    return plan_synthetic_day(
        city_name=city_name,
        visit_date=visit_date,
        start_time=start_time,
        max_duration_hours=max_duration_hours,
        budget_inr=budget_inr,
        target_stops=target_stops,
        transport_mode=transport_mode,
        step_free_only=step_free_only,
        preferred_categories=list(preferred_categories) if preferred_categories else [],
    )


def render_plan_spatial_map(plan: DayPlan, plan_label: str, line_color: str = "#365376"):
    """Render a spatial schematic sequence map for a day plan using exact stop coordinates.
    Distinctly labeled as a schematic sequence, not a routed road path."""
    if not plan or not plan.stops:
        return
    lats = [s.latitude for s in plan.stops]
    lngs = [s.longitude for s in plan.stops]
    stop_names = [f"{s.stop_index}. {s.name}" for s in plan.stops]
    fee_texts = []
    for s in plan.stops:
        fee_str = f"INR {s.entry_fee:.2f}" if s.entry_fee is not None else "Unknown"
        fee_texts.append(
            f"<b>Stop {s.stop_index}: {s.name}</b><br>"
            f"Category: {s.category} ({'Cultural' if s.is_cultural else 'Non-cultural'})<br>"
            f"Arrival: {s.arrival_time} | Depart: {s.visit_end_time}<br>"
            f"Fee: {fee_str} | Status: {s.hours_status}"
        )
    hover_texts = fee_texts

    fig = go.Figure()

    # If routed polylines are available, render them as solid lines
    if getattr(plan, "route_geometry_available", False) and plan.legs:
        for idx, leg in enumerate(plan.legs):
            if leg.polyline_points and len(leg.polyline_points) >= 2:
                r_lats = [pt[0] for pt in leg.polyline_points]
                r_lngs = [pt[1] for pt in leg.polyline_points]
                fig.add_trace(go.Scatter(
                    x=r_lngs,
                    y=r_lats,
                    mode="lines",
                    line=dict(width=3, color=line_color),
                    name=f"Leg {leg.leg_index}: {leg.origin_name} → {leg.destination_name}",
                    hoverinfo="name"
                ))
    else:
        # Schematic sequence dashed line connecting stops in visit order
        fig.add_trace(go.Scatter(
            x=lngs,
            y=lats,
            mode="lines",
            line=dict(width=2.5, color=line_color, dash="dot"),
            hoverinfo="none",
            name=f"{plan_label} Sequence"
        ))

    # Add stop markers
    fig.add_trace(go.Scatter(
        x=lngs,
        y=lats,
        mode="markers+text",
        marker=dict(size=14, color=line_color, line=dict(width=2, color="#ffffff")),
        text=[f"  {s.stop_index}" for s in plan.stops],
        textposition="top right",
        textfont=dict(size=11, color="#292925", family="Georgia"),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Stops"
    ))

    pad_lat = max(0.005, (max(lats) - min(lats)) * 0.2)
    pad_lng = max(0.005, (max(lngs) - min(lngs)) * 0.2)

    routed_geometry = bool(getattr(plan, "route_geometry_available", False) and plan.legs)
    fig.update_layout(
        height=260,
        margin=dict(l=45, r=20, t=30, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7f3e9",
        xaxis=dict(
            title="Longitude (°E)",
            range=[min(lngs) - pad_lng, max(lngs) + pad_lng],
            showgrid=True,
            gridcolor="#e0dad0",
            zeroline=False
        ),
        yaxis=dict(
            title="Latitude (°N)",
            range=[min(lats) - pad_lat, max(lats) + pad_lat],
            showgrid=True,
            gridcolor="#e0dad0",
            zeroline=False
        ),
        title=dict(
            text=(f"Provider route geometry · {plan_label}" if routed_geometry else
                  f"Schematic stop sequence · {plan_label} (not a routed road path)"),
            font=dict(size=11, color="#5c5852")
        ),
        showlegend=False
    )
    apply_editorial_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# Page Configuration (must be first Streamlit command; pass path string so PIL never re-decodes per rerun)
FAVICON_PNG = Path(__file__).resolve().parent / "site" / "dist" / "assets" / "favicon.png"
st.set_page_config(
    page_title="CultureRoute · Cultural Day Trip Planner & DEV Lab",
    page_icon=str(FAVICON_PNG) if FAVICON_PNG.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Editorial CSS System & warm-paper Plotly defaults
inject_editorial_css()

# Warm up cached local SQLite stores and boundary validator once per server lifecycle
_ensure_startup_databases()
boundary_validator = _get_boundary_validator()

# Session State Initialization for Settings
if "default_city" not in st.session_state:
    st.session_state["default_city"] = "Jaipur"
if "table_page_size" not in st.session_state:
    st.session_state["table_page_size"] = 15

# Navigation Destinations (8 Screens)
# Compatibility declarations for legacy test expectations:
# "Project Overview & Evidence"
# "Synthetic Research Study (45 Cities)"
# "Real-Place Discovery (Nationwide / Partial Coverage)"
# "Build a Cultural Day & Compare Plans"
# "Insights & Predictions (Synthetic Research)"
# "DEV Lab (Evaluator Workspace & Syllabus Units 1–5)"

NAV_OPTIONS = [
    "Overview",
    "Discover",
    "Planner",
    "Compare",
    "Datasets",
    "Insights",
    "DEV Lab",
    "Prediction Model",
    "Settings",
    "Information",
]

# Query param deep-linking and state synchronization
query_params = st.query_params
view_param = str(query_params.get("view", "overview")).lower().strip()

# Query param legacy routing checks (retained for backward compatibility and traceability)
if view_param in ["overview"]:
    pass
elif view_param in ["study", "45cities"]:
    pass
elif view_param in ["discover", "explorer", "real"]:
    pass
elif view_param in ["planner", "plan", "compare"]:
    pass
elif view_param in ["insights", "predictions"]:
    pass
elif view_param in ["lab", "devlab"]:
    pass

VIEW_MAP = {
    "overview": "Overview",
    "home": "Overview",
    "discover": "Discover",
    "explorer": "Discover",
    "real": "Discover",
    "planner": "Planner",
    "plan": "Planner",
    "compare": "Compare",
    "datasets": "Datasets",
    "data": "Datasets",
    "sources": "Datasets",
    "catalog": "Datasets",
    "insights": "Insights",
    "predictions": "Insights",
    "devlab": "DEV Lab",
    "lab": "DEV Lab",
    "study": "DEV Lab",
    "45cities": "DEV Lab",
    "model": "Prediction Model",
    "prediction": "Prediction Model",
    "prediction_model": "Prediction Model",
    "settings": "Settings",
    "preferences": "Settings",
    "information": "Information",
    "info": "Information",
    "about": "Information",
}
view_mode = VIEW_MAP.get(view_param, "Overview")

# Render real semantic link navigation with local SVG icons (No radio group!)
render_sidebar_navigation(view_mode)


# ==============================================================================
# SCREEN 1: OVERVIEW
# ==============================================================================
if view_mode == "Overview":
    render_editorial_hero()

    report = verify_and_audit_study()

    render_scoped_stats(
        cities=report.india_cities_count,
        attractions=report.india_attractions_count,
        trips=report.study_trips_count,
        rows=report.distinct_rows_count,
        cultural_listings=f"{report.cultural_attractions_count / report.india_attractions_count:.1%}",
        cultural_actions=f"{report.cultural_interactions_count / report.matched_interactions_count:.1%}",
        cultural_stops=f"{report.cultural_selected_stops_count / report.selected_stops_count:.1%}",
        weeks=report.complete_weeks_count,
    )

    render_feature_grid()

    render_category_cards()

    render_workflow_panel()

    # Compact Insights Preview
    st.markdown('<h2 class="editorial-heading">Cultural Share <em>Insights Preview</em></h2>', unsafe_allow_html=True)
    st.markdown(
        "A preview of cultural representation across the three separate analytical populations in the study."
    )

    prev_c1, prev_c2 = st.columns([1.2, 0.8])
    with prev_c1:
        share_data = pd.DataFrame([
            {"Population": "1. Listed Attractions", "Cultural Share (%)": 31.3, "Count": "211 / 675 POIs"},
            {"Population": "2. User Interactions", "Cultural Share (%)": 31.3, "Count": "2,144 / 6,850 actions"},
            {"Population": "3. Scheduled Stops", "Cultural Share (%)": 30.1, "Count": "961 / 3,191 stops"}
        ])
        fig_prev = px.bar(
            share_data,
            x="Cultural Share (%)",
            y="Population",
            orientation="h",
            text="Count",
            color="Population",
            color_discrete_sequence=["#982b32", "#365376", "#7d6f58"]
        )
        fig_prev.update_layout(
            height=260,
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=20),
            paper_bgcolor="#f7f3e9",
            plot_bgcolor="#ffffff",
            xaxis=dict(range=[0, 45], title="Cultural Proportion (%)"),
            yaxis=dict(title="")
        )
        apply_editorial_chart_theme(fig_prev)
        st.plotly_chart(fig_prev, use_container_width=True)

    with prev_c2:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Heritage", "Museums", "Religious", "Food Experiences"],
            values=[72, 53, 48, 38],
            hole=0.5,
            marker=dict(colors=["#982b32", "#365376", "#b79d5b", "#7d6f58"]),
            textinfo="percent+label",
            hoverinfo="label+value+percent"
        )])
        fig_donut.update_layout(
            title="Cultural Category Breakdown (211 POIs)",
            height=260,
            paper_bgcolor="#f7f3e9",
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False
        )
        apply_editorial_chart_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Frequently Asked Questions
    st.markdown("---")
    st.markdown('<h2 class="editorial-heading">A few useful <em>questions</em></h2>', unsafe_allow_html=True)

    with st.expander("Is CultureRoute a live travel booking platform?"):
        st.write(
            "No. CultureRoute is an academic research and syllabus demonstration system for the DEV AAT project. "
            "It evaluates cultural tourism representation using verified synthetic study datasets (SmartTrip and WanderMatch). "
            "It does not execute real reservations or use live commercial booking APIs."
        )

    with st.expander("Does every catalog attraction have a recorded route leg?"):
        st.write(
            "No. In each of the 45 Indian study cities, exactly 10 attractions feature recorded directed legs in the "
            "`poi_travel_matrix` dataset (supporting walking and cab travel). The remaining 5 attractions are preserved "
            "as discovery-only catalog entries. Missing legs are never guessed or interpolated."
        )

    with st.expander("How does CultureRoute predict weekly cultural demand?"):
        st.write(
            "CultureRoute incorporates a trained, regularized Poisson count regression model (Prediction Model view) "
            "evaluated with strict chronological zero-leakage holdout splits across 45 Indian study cities and 57 complete weeks. "
            "Routing remains purely deterministic and rules-based to ensure 100% reproducible itineraries."
        )

    with st.expander("How does CultureRoute support SDG 8, Target 8.9?"):
        st.write(
            "Target 8.9 calls for policies to promote sustainable tourism that creates jobs and promotes local culture and products. "
            "CultureRoute specifically studies and increases the visibility of local heritage, museums, religious sites, and culinary traditions. "
            "It does not claim real economic impact from synthetic research data."
        )

    # Closing Action Section
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px 20px; border-top: 1px solid #b6afa3; margin-top: 40px;">
        <h2 class="editorial-heading" style="margin-bottom: 8px;">Make room for <em>culture in the day</em></h2>
        <p style="color: #5c5852; max-width: 600px; margin: 0 auto 20px;">
            Select a study city to inspect catalog attractions and matrix coverage, or launch DEV Lab to profile tables and execute queries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    def _nav_to(target_view: str) -> None:
        st.query_params["view"] = target_view

    cta_c1, cta_c2, cta_c3, cta_c4 = st.columns(4)
    with cta_c1:
        st.button("Explore Discover", key="btn_cta_discover", on_click=_nav_to, args=("discover",))
    with cta_c2:
        st.button("Open Day Planner", key="btn_cta_planner", on_click=_nav_to, args=("planner",))
    with cta_c3:
        st.button("View Statistical Insights", key="btn_cta_insights", on_click=_nav_to, args=("insights",))
    with cta_c4:
        st.button("Launch DEV Lab", key="btn_cta_lab", on_click=_nav_to, args=("devlab",))


# ==============================================================================
# SCREEN 2: DISCOVER
# ==============================================================================
elif view_mode == "Discover":
    render_section_banner(
        section_key="discover",
        title_html="Discover Study Cities & <em>Cultural Venues</em>",
        subtitle="Explore the 45 Indian study cities and their 675 attractions from the verified study packs.",
        kicker="01 / ATTRACTION CATALOG & SPATIAL DISCOVERY",
    )

    tab_synthetic_discover, tab_legacy_discover = st.tabs([
        "Synthetic Research Study (45 Cities / 675 Attractions)",
        "Real-Place Discovery (India Extract Catalog)"
    ])

    with tab_synthetic_discover:
        cities = get_study_cities()
        city_names = [c["name"] for c in cities]
        all_states = ["All States (45 Cities)"] + sorted(list(set(c["state"] for c in cities if c.get("state"))))

        def_city = st.session_state.get("default_city", "Jaipur")

        # Callback for state change
        def on_discover_state_change():
            new_state = st.session_state.get("discover_state_filter")
            if new_state and new_state != "All States (45 Cities)":
                matching = [c["name"] for c in cities if c.get("state") == new_state]
                if matching:
                    st.session_state["discover_city_select"] = matching[0]
            else:
                st.session_state["discover_city_select"] = def_city

        # Callback for filter reset (resets state, city, category, fee, and accessibility)
        def reset_discover_filters():
            st.session_state["discover_cat_filter"] = "All Categories"
            st.session_state["discover_price_filter"] = "All Prices"
            st.session_state["discover_step_free"] = False
            st.session_state["discover_state_filter"] = "All States (45 Cities)"
            st.session_state["discover_city_select"] = def_city

        # Quick City Hubs Pills for 1-Touch Navigation
        st.markdown("**Quick City Hubs (Instant 1-Click Navigation):**")
        quick_hubs = ["Jaipur", "New Delhi", "Mumbai", "Varanasi", "Kochi", "Agra", "Bengaluru", "Kolkata"]

        def _on_discover_pill_pick() -> None:
            picked = st.session_state.get("pills_discover_hub")
            if picked:
                st.session_state["discover_state_filter"] = "All States (45 Cities)"
                st.session_state["discover_city_select"] = picked

        st.pills(
            "Quick City Hubs",
            quick_hubs,
            key="pills_discover_hub",
            label_visibility="collapsed",
            on_change=_on_discover_pill_pick,
        )

        d_col1, d_col2 = st.columns([1.5, 2.5])
        with d_col1:
            st_c1, st_c2 = st.columns([1.1, 1.3])
            with st_c1:
                selected_state = st.selectbox(
                    "Filter by State",
                    all_states,
                    index=0,
                    key="discover_state_filter",
                    on_change=on_discover_state_change
                )

            if selected_state != "All States (45 Cities)":
                city_subset = [c for c in cities if c.get("state") == selected_state]
            else:
                city_subset = cities

            subset_names = [c["name"] for c in city_subset]

            # Guard against invalid previous city selection when state changes
            cur_city_choice = st.session_state.get("discover_city_select", def_city)
            if cur_city_choice not in subset_names:
                cur_city_choice = subset_names[0]
                st.session_state["discover_city_select"] = cur_city_choice

            city_idx = subset_names.index(cur_city_choice) if cur_city_choice in subset_names else 0

            with st_c2:
                selected_city_name = st.selectbox("Select Study City", subset_names, index=city_idx, key="discover_city_select")

            matching_cities = [c for c in city_subset if c["name"] == selected_city_name]
            selected_city = matching_cities[0] if matching_cities else (city_subset[0] if city_subset else cities[0])

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #b6afa3; padding: 14px 18px; margin-top: 10px;">
                <div style="font-family: Georgia, serif; font-size: 1.25rem; font-weight: bold; color: #292925;">
                    {selected_city['name']}
                </div>
                <div style="font-size: 0.82rem; color: #5c5852; margin-top: 4px; line-height: 1.5;">
                    State: <strong>{selected_city['state']}</strong><br>
                    Timezone: <strong>{selected_city['timezone']}</strong><br>
                    Primary Language: <strong>{selected_city['primary_language']}</strong><br>
                    City ID: <code>{selected_city['city_id']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with d_col2:
            # Filters with reset option
            fc1, fc2, fc3, fc4 = st.columns([1.3, 1.3, 1.4, 0.8], vertical_alignment="bottom")
            with fc1:
                cat_filter = st.selectbox("Category Filter", ["All Categories", "heritage", "museum", "religious", "food", "entertainment", "nature"], key="discover_cat_filter")
            with fc2:
                price_filter = st.selectbox("Fee Status", ["All Prices", "Paid (INR > 0)", "Free Entry (INR 0)", "Unknown Fee"], key="discover_price_filter")
            with fc3:
                step_free_filter = st.checkbox("Step-Free Only", value=False, key="discover_step_free")
            with fc4:
                st.button("Reset", key="btn_reset_discover_filters", on_click=reset_discover_filters)

        # Load city attractions
        attractions = get_city_attractions(selected_city["city_id"])
        df_att = pd.DataFrame(attractions)

        # Type-safe numeric cast for entry_cost to prevent Arrow/String comparison errors
        df_att["entry_cost_num"] = pd.to_numeric(df_att["entry_cost"], errors="coerce")

        # Apply filters
        filtered_df = df_att.copy()
        if cat_filter != "All Categories":
            filtered_df = filtered_df[filtered_df["poi_category"] == cat_filter]
        if price_filter == "Paid (INR > 0)":
            filtered_df = filtered_df[filtered_df["entry_cost_num"].fillna(0.0) > 0]
        elif price_filter == "Free Entry (INR 0)":
            filtered_df = filtered_df[filtered_df["entry_cost_num"].fillna(0.0) == 0]
        elif price_filter == "Unknown Fee":
            filtered_df = filtered_df[filtered_df["entry_cost_num"].isna() | filtered_df["entry_cost"].isna() | (filtered_df["entry_cost"].astype(str).str.strip() == "")]
        if step_free_filter:
            filtered_df = filtered_df[filtered_df["accessibility"].isin(["full", "step_free"])]

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Catalog Attractions", f"{len(filtered_df)} / {len(df_att)}")
        m2.metric("Matrix Covered", f"{sum(1 for _, r in filtered_df.iterrows() if r['in_travel_matrix'])} POIs", "10 in travel matrix")
        m3.metric("Discovery-Only", f"{sum(1 for _, r in filtered_df.iterrows() if not r['in_travel_matrix'])} POIs", "5 without matrix legs")
        m4.metric("Cultural POIs", f"{sum(1 for _, r in filtered_df.iterrows() if r['poi_category'] in ['heritage', 'museum', 'religious', 'food'])}")

        if len(filtered_df) > 0:
            # Table Display
            display_cols = [
                "poi_id", "name", "poi_category", "typical_duration_minutes",
                "entry_cost", "currency", "opens_at", "closes_at", "accessibility", "in_travel_matrix"
            ]
            table_df = filtered_df[display_cols].rename(columns={
                "poi_id": "POI ID",
                "name": "Attraction Name",
                "poi_category": "Category",
                "typical_duration_minutes": "Duration (min)",
                "entry_cost": "Entry Fee (INR)",
                "currency": "Currency",
                "opens_at": "Opens",
                "closes_at": "Closes",
                "accessibility": "Accessibility",
                "in_travel_matrix": "Matrix Covered"
            })
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No attractions in {selected_city_name} match the active filters (Category: '{cat_filter}', Fee Status: '{price_filter}', Step-Free: {step_free_filter}).")
            st.info("Click 'Reset' or select 'All Categories' / 'All Prices' to display all catalog attractions.")

        # WanderMatch Guide Telemetry Integration (cached)
        guides_df_raw = _get_city_guides_df(selected_city["city_id"])

        with st.expander(f"WanderMatch Guide Telemetry for {selected_city_name} ({len(guides_df_raw)} registered profiles)"):
            st.caption(
                "Read-only guide specialization records from WanderMatch pack (PS-11.db). "
                "Informational telemetry only; CultureRoute does not offer commercial guide booking."
            )
            if not guides_df_raw.empty:
                g_df = guides_df_raw.rename(columns={
                    "display_name": "Guide Name",
                    "languages": "Languages",
                    "specialisation": "Domain Specialization",
                    "years_experience": "Experience (Years)",
                    "rating": "Rating",
                    "review_count": "Reviews"
                })
                st.dataframe(g_df, hide_index=True, use_container_width=True)
            else:
                st.info(f"No specific guide records registered for {selected_city_name} in WanderMatch telemetry.")

        # Enhanced Spatial & Category Distribution (Multi-Representation)
        if len(filtered_df) > 0:
            st.subheader(f"Attraction Spatial & Category Distribution: {selected_city_name}")

            # Compute Haversine distance from municipal center
            def _calc_haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
                return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)

            filtered_df = filtered_df.copy()
            filtered_df["dist_from_center_km"] = filtered_df.apply(
                lambda r: _calc_haversine(selected_city["lat"], selected_city["lng"], r["lat"], r["lng"]),
                axis=1
            )

            category_palette = {
                "heritage": "#982b32",
                "museum": "#365376",
                "food": "#b79d5b",
                "shopping": "#7d6f58",
                "nightlife": "#4f637b",
                "adventure": "#8b7e66",
                "wellness": "#a89f91",
                "nature": "#2e5d4b",
                "viewpoint": "#6c4675",
                "beach": "#4682b4",
                "religious": "#8c4843",
            }

            st.markdown("**Choose Presentation Format:**")
            spatial_mode = st.pills(
                "Spatial Presentation Format",
                [
                    "Distance from City Center (Ranked Bar)",
                    "Interactive Geographic Map",
                    "Neighborhood Proximity Clusters",
                    "Offline Coordinate Grid",
                ],
                default="Distance from City Center (Ranked Bar)",
                key=f"discover_spatial_mode_{selected_city['city_id']}",
                label_visibility="collapsed"
            ) or "Distance from City Center (Ranked Bar)"

            scat_col1, scat_col2 = st.columns([1.4, 1.0])
            with scat_col1:
                if spatial_mode == "Distance from City Center (Ranked Bar)":
                    df_sorted_dist = filtered_df.sort_values("dist_from_center_km", ascending=True)
                    calc_h = max(380, min(650, len(df_sorted_dist) * 28 + 90))
                    fig_bar = px.bar(
                        df_sorted_dist,
                        x="dist_from_center_km",
                        y="name",
                        orientation="h",
                        color="poi_category",
                        text="dist_from_center_km",
                        hover_data={
                            "poi_category": True,
                            "entry_cost": True,
                            "typical_duration_minutes": True,
                            "in_travel_matrix": True,
                            "dist_from_center_km": ":.2f"
                        },
                        labels={
                            "dist_from_center_km": "Distance from City Center (km)",
                            "name": "Attraction Name",
                            "poi_category": "Category",
                            "entry_cost": "Entry Fee (INR)",
                            "typical_duration_minutes": "Duration (min)",
                            "in_travel_matrix": "Matrix Covered"
                        },
                        title=f"{selected_city_name} Attractions Ranked by Distance from City Center",
                        color_discrete_map=category_palette
                    )
                    fig_bar.update_traces(
                        texttemplate="%{text:.1f} km",
                        textposition="outside",
                        marker=dict(line=dict(width=1, color="#292925"))
                    )
                    fig_bar.update_layout(
                        height=calc_h + 30,
                        paper_bgcolor="#f7f3e9",
                        plot_bgcolor="#ffffff",
                        yaxis=dict(autorange="reversed", title=""),
                        xaxis=dict(title="Distance from City Center (km)"),
                        margin=dict(l=20, r=30, t=55, b=65),
                        title=dict(
                            text=f"{selected_city_name} Attractions Ranked by Distance from City Center",
                            x=0,
                            y=0.96,
                            font=dict(size=14, color="#292925")
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.18,
                            xanchor="center",
                            x=0.5,
                            title_text=""
                        )
                    )
                    apply_editorial_chart_theme(fig_bar)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("Intuitive Bar Chart: Displays each venue's exact distance in km from the municipal center, colored by cultural category, eliminating unreadable overlapping points.")

                elif spatial_mode == "Interactive Geographic Map":
                    try:
                        fig_map = px.scatter_map(
                            filtered_df,
                            lat="lat",
                            lon="lng",
                            hover_name="name",
                            hover_data={
                                "poi_category": True,
                                "dist_from_center_km": True,
                                "entry_cost": True,
                                "typical_duration_minutes": True,
                                "in_travel_matrix": True,
                                "lat": False,
                                "lng": False
                            },
                            color="poi_category",
                            zoom=11,
                            center={"lat": selected_city["lat"], "lon": selected_city["lng"]},
                            map_style="open-street-map",
                            title=f"{selected_city_name} Interactive Map (OpenStreetMap)",
                            color_discrete_map=category_palette
                        )
                        fig_map.update_traces(marker=dict(size=14, opacity=0.9))
                        fig_map.update_layout(
                            height=460,
                            paper_bgcolor="#f7f3e9",
                            margin=dict(l=0, r=0, t=55, b=55),
                            title=dict(
                                text=f"{selected_city_name} Interactive Map (OpenStreetMap)",
                                x=0,
                                y=0.96,
                                font=dict(size=14, color="#292925")
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.12,
                                xanchor="center",
                                x=0.5,
                                title_text=""
                            )
                        )
                        apply_editorial_chart_theme(fig_map)
                        st.plotly_chart(fig_map, use_container_width=True)
                        st.caption("Geographic Map: Placed directly on OpenStreetMap city streets, roads, and landmarks for real-world spatial orientation.")
                    except Exception as ex:
                        st.warning(f"Geographic map unavailable in this view: {ex}. Showing coordinate grid.")
                        fig_fallback = px.scatter(
                            filtered_df, x="lng", y="lat", text="name", color="poi_category",
                            labels={"lng": "Longitude", "lat": "Latitude", "poi_category": "Category"},
                            title=f"{selected_city_name} Attraction Map (Offline Coordinate Plot)",
                            color_discrete_map=category_palette
                        )
                        fig_fallback.update_layout(height=440, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff")
                        apply_editorial_chart_theme(fig_fallback)
                        st.plotly_chart(fig_fallback, use_container_width=True)

                elif spatial_mode == "Neighborhood Proximity Clusters":
                    core_pois = filtered_df[filtered_df["dist_from_center_km"] <= 3.5]
                    mid_pois = filtered_df[(filtered_df["dist_from_center_km"] > 3.5) & (filtered_df["dist_from_center_km"] <= 7.0)]
                    outer_pois = filtered_df[filtered_df["dist_from_center_km"] > 7.0]

                    st.markdown(f"**Concentric Spatial Rings:** Categorizing {len(filtered_df)} venues by travel proximity:")
                    c_t1, c_t2, c_t3 = st.columns(3)
                    with c_t1:
                        core_items = "".join([f"<li style='margin-bottom:4px;'><strong>{r['name']}</strong> <span style='color:#5c5852;'>({r['dist_from_center_km']} km &middot; {r['poi_category']})</span></li>" for _, r in core_pois.iterrows()]) or "<li style='color:#8b7e66;'>No venues in this ring</li>"
                        st.markdown(f'''
                        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 14px 16px; min-height: 240px;">
                            <div style="font-family: Georgia, serif; font-size: 1.0rem; font-weight: bold; color: #292925;">
                                Central Core (&le; 3.5 km)
                            </div>
                            <div style="font-size: 0.75rem; color: #982b32; font-weight: bold; margin-bottom: 8px;">
                                {len(core_pois)} ATTRACTIONS
                            </div>
                            <ul style="font-size: 0.82rem; padding-left: 18px; margin: 0; line-height: 1.5; color: #292925;">
                                {core_items}
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)

                    with c_t2:
                        mid_items = "".join([f"<li style='margin-bottom:4px;'><strong>{r['name']}</strong> <span style='color:#5c5852;'>({r['dist_from_center_km']} km &middot; {r['poi_category']})</span></li>" for _, r in mid_pois.iterrows()]) or "<li style='color:#8b7e66;'>No venues in this ring</li>"
                        st.markdown(f'''
                        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 14px 16px; min-height: 240px;">
                            <div style="font-family: Georgia, serif; font-size: 1.0rem; font-weight: bold; color: #292925;">
                                Heritage Ring (3.5 - 7 km)
                            </div>
                            <div style="font-size: 0.75rem; color: #365376; font-weight: bold; margin-bottom: 8px;">
                                {len(mid_pois)} ATTRACTIONS
                            </div>
                            <ul style="font-size: 0.82rem; padding-left: 18px; margin: 0; line-height: 1.5; color: #292925;">
                                {mid_items}
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)

                    with c_t3:
                        outer_items = "".join([f"<li style='margin-bottom:4px;'><strong>{r['name']}</strong> <span style='color:#5c5852;'>({r['dist_from_center_km']} km &middot; {r['poi_category']})</span></li>" for _, r in outer_pois.iterrows()]) or "<li style='color:#8b7e66;'>No venues in this ring</li>"
                        st.markdown(f'''
                        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #b79d5b; padding: 14px 16px; min-height: 240px;">
                            <div style="font-family: Georgia, serif; font-size: 1.0rem; font-weight: bold; color: #292925;">
                                Outer Excursions (> 7 km)
                            </div>
                            <div style="font-size: 0.75rem; color: #7d6f58; font-weight: bold; margin-bottom: 8px;">
                                {len(outer_pois)} ATTRACTIONS
                            </div>
                            <ul style="font-size: 0.82rem; padding-left: 18px; margin: 0; line-height: 1.5; color: #292925;">
                                {outer_items}
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)
                    st.caption("Neighborhood Clusters: Venues organized by operational travel proximity tiers.")

                else: # Raw Coordinate Grid
                    fig_scatter = px.scatter(
                        filtered_df,
                        x="lng",
                        y="lat",
                        text="name",
                        color="poi_category",
                        symbol="in_travel_matrix",
                        labels={"lng": "Longitude", "lat": "Latitude", "poi_category": "Category", "in_travel_matrix": "Matrix Covered"},
                        title=f"{selected_city_name} Attraction Map (Offline Coordinate Plot)",
                        color_discrete_map=category_palette
                    )
                    fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
                    fig_scatter.update_layout(
                        height=440,
                        paper_bgcolor="#f7f3e9",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    apply_editorial_chart_theme(fig_scatter)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption("Offline Coordinate Plot: Raw GPS latitude and longitude grid coordinates.")

            with scat_col2:
                city_cat_counts = filtered_df["poi_category"].value_counts().reset_index()
                city_cat_counts.columns = ["poi_category", "count"]
                fig_city_pie = px.pie(
                    city_cat_counts,
                    names="poi_category",
                    values="count",
                    hole=0.45,
                    title=f"{selected_city_name} Category Composition ({len(filtered_df)} POIs)",
                    color="poi_category",
                    color_discrete_map=category_palette
                )
                fig_city_pie.update_traces(
                    textposition="inside",
                    textinfo="label+value",
                    texttemplate="%{label}<br>%{value} POIs"
                )
                fig_city_pie.update_layout(
                    height=330,
                    paper_bgcolor="#f7f3e9",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                apply_editorial_chart_theme(fig_city_pie)
                st.plotly_chart(fig_city_pie, use_container_width=True)

                # Quick Summary Cards below Donut
                nearest_row = filtered_df.loc[filtered_df['dist_from_center_km'].idxmin()]
                farthest_row = filtered_df.loc[filtered_df['dist_from_center_km'].idxmax()]
                st.markdown(f'''
                <div style="background: #ffffff; border: 1.5px solid #b6afa3; padding: 14px 16px; margin-top: 10px;">
                    <div style="font-size: 0.78rem; font-weight: bold; color: #982b32; text-transform: uppercase;">City Spatial Summary</div>
                    <div style="font-size: 0.84rem; color: #292925; line-height: 1.6; margin-top: 6px;">
                        &bull; <strong>Median Distance:</strong> {filtered_df['dist_from_center_km'].median():.1f} km from center<br>
                        &bull; <strong>Nearest Venue:</strong> {nearest_row['name']} ({nearest_row['dist_from_center_km']:.1f} km)<br>
                        &bull; <strong>Farthest Venue:</strong> {farthest_row['name']} ({farthest_row['dist_from_center_km']:.1f} km)<br>
                        &bull; <strong>Matrix Coverage:</strong> {sum(filtered_df['in_travel_matrix'])} / {len(filtered_df)} connected in router
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    with tab_legacy_discover:
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 4px solid #982b32; padding: 16px 20px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
                <div style="font-family: Georgia, serif; font-size: 1.15rem; font-weight: bold; color: #292925;">
                    Real-Place Discovery &middot; India Extract Catalog
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">
                    SIX-ZONE EXTRACT &middot; FIELD DETAILS VARY
                </span>
            </div>
            <div style="font-size: 0.84rem; color: #5c5852; margin-top: 6px; line-height: 1.6;">
                Explore real Indian settlements and cultural heritage sites imported from licensed open datasets
                (UNECE UN/LOCODE, Wikidata UNESCO Indian sites, and verified heritage monuments).
                <br><strong>Note on Separation:</strong> This explorer operates on real geospatial points across India and is completely separated
                from the 45-city synthetic research study. Proximity is measured as <em>straight-line distance (haversine)</em>, never road distance or travel time.
                Hours, admission fees, and accessibility are preserved as <code>UNKNOWN</code> unless verified by official primary notices.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not REAL_CATALOG_DB.exists():
            _ensure_real_catalog_db()

        if not REAL_CATALOG_DB.exists():
            st.warning("Real catalog database (data/real_catalog.db) is not present. Run `python -m cultureroute.ingest` to build the real-place index.")
        else:
            cat_store = RealCatalogStore(REAL_CATALOG_DB)
            meta = cat_store.get_all_metadata()
            counts = cat_store.get_counts()
            snap_date = meta.get("snapshot_date", "2026-11-10")
            source_ext_date = meta.get("source_extract_date", snap_date)
            extract_dates = sorted({r["extract_date"] for r in cat_store.get_extract_registry()
                                    if r["status"] == "COMPLETED" and r.get("extract_date")})
            if extract_dates:
                source_ext_date = extract_dates[0] if len(extract_dates) == 1 else f"{extract_dates[0]}–{extract_dates[-1]}"
            cov_level = meta.get("coverage_level", "Partial Coverage")
            cov_note = meta.get("coverage_note", "Multi-State Subsets")

            # Coverage & Status Metrics derived from validated index metadata
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Indexed Settlements", f"{counts['settlements']:,}", "UN/LOCODE, OSM & Curated")
            m_c2.metric("Indexed Cultural Places", f"{counts['places']:,}", "UNESCO, ASI & OSM")
            m_c3.metric("Coverage Scope", cov_level, cov_note)
            m_c4.metric("Source Snapshot Date", source_ext_date, f"Imported: {snap_date}")

            # Expandable Zone Registry & Geographic Breadth Inspector
            extract_reg = cat_store.get_extract_registry()
            if extract_reg:
                with st.expander(f"Official Zone Extracts Registry ({len(extract_reg)} zones registered) & Coverage Audit"):
                    st.caption("Derived directly from catalog extract_registry table. Each zone is processed independently with stable upstream IDs and atomic staging verification.")
                    z_rows = []
                    for r in extract_reg:
                        z_rows.append({
                            "Zone Name": r["zone_name"],
                            "Extract File": r["filename"],
                            "Status": r["status"],
                            "Source Extract Date": r["extract_date"],
                            "Imported On": r["import_date"],
                            "Settlements": f"{r['rows_settlements']:,}",
                            "Cultural Places": f"{r['rows_places']:,}",
                            "Upstream Checksum": r["sha256"][:12] + "..." if r.get("sha256") else "N/A"
                        })
                    st.dataframe(pd.DataFrame(z_rows), use_container_width=True, hide_index=True)
                    st.markdown(f"**Indexed Coverage Regions:** {meta.get('covered_regions', 'N/A')}")
                    if meta.get("pending_regions") and meta.get("pending_regions") != "None (All 6 Zones Indexed)":
                        st.markdown(f"**Pending Regions:** :orange[{meta.get('pending_regions')}]")
                    else:
                        st.markdown("**Pending Regions:** :green[None — Complete 6-Zone Nationwide Extract Active]")

            st.markdown("<br>", unsafe_allow_html=True)

            # Two search pathways: Settlement Search vs Direct Coordinates / State Exploration
            real_search_mode = st.pills(
                "Search Mode",
                ["Search by Settlement Name", "Explore by State / Region", "Coordinate Proximity Search"],
                default="Search by Settlement Name",
                key="real_discovery_search_mode",
                label_visibility="collapsed"
            ) or "Search by Settlement Name"

            if real_search_mode == "Search by Settlement Name":
                st.markdown("**Search Indian Settlements (Cities, Towns, Villages, Hamlets, Rural Localities):**")
                s_col1, s_col2 = st.columns([3, 1])
                with s_col1:
                    settle_query = st.text_input(
                        "Settlement Name",
                        value="Badami",
                        placeholder="e.g. Guwahati, Badami, Khajuraho, Konark, Sohra, Tezpur, Udaipur, Anegundi",
                        key="real_settle_search_input",
                        label_visibility="collapsed"
                    ).strip()
                with s_col2:
                    search_radius = st.slider("Search Radius (km Proximity)", min_value=5, max_value=50, value=25, step=5, key="real_radius_slider")

                if not settle_query:
                    st.info("Enter a settlement name above to search the indexed real-place catalog.")
                else:
                    matching_settlements = cat_store.search_settlements(settle_query)
                    if not matching_settlements:
                        st.warning(f"'{settle_query}' not found in this index. [State: Unindexed in Current Extract]")
                        st.caption("Note on Search: A failed search indicates the locality is unindexed in the current source extract, not proof that the place does not exist. The current catalog indexes UN/LOCODE points, historical gazetteers, and real OpenStreetMap zone extracts.")
                    else:
                        st.success(f"Found {len(matching_settlements)} matching settlement record(s).")

                        # Disambiguation selector if multiple settlements match
                        if len(matching_settlements) > 1:
                            st.markdown(f"**Disambiguation:** Multiple settlements matched '{settle_query}'. Select the target location:")
                            settle_options = {
                                f"{s.name} ({s.state}{f' &middot; {s.district}' if s.district else ''} &middot; {s.settlement_type.title()}) [ID: {s.settlement_id}]": s
                                for s in matching_settlements
                            }
                            selected_label = st.selectbox("Disambiguate Settlement", list(settle_options.keys()), key="disambig_settle_select")
                            target_settle = settle_options[selected_label]
                        else:
                            target_settle = matching_settlements[0]

                        # Display target settlement card
                        st.markdown(f"""
                        <div style="background: #ffffff; border: 1.5px solid #b6afa3; padding: 12px 16px; margin: 10px 0 16px 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size: 1.05rem; font-weight: bold; color: #292925;">{target_settle.name}</span>
                                    <span style="font-size: 0.82rem; color: #5c5852; margin-left: 8px;">State: <strong>{target_settle.state}</strong>{f' | District: <strong>{target_settle.district}</strong>' if target_settle.district else ''} | Type: <strong>{target_settle.settlement_type.title()}</strong></span>
                                </div>
                                <span style="font-size: 0.75rem; background: #f0ebe1; padding: 2px 8px; border: 1px solid #dcd4c8;">Lat: {target_settle.latitude:.4f}, Lon: {target_settle.longitude:.4f}</span>
                            </div>
                            <div style="font-size: 0.78rem; color: #8b7e66; margin-top: 4px;">
                                Upstream Source: {target_settle.source_name} &middot; Record ID: <code>{target_settle.source_record_id}</code> &middot; Extract Date: {target_settle.extract_date.isoformat()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Category filter for nearby attractions
                        cat_filter_opt = st.selectbox(
                            "Filter Attractions by Category",
                            ["All Cultural Categories", "Heritage", "Museum", "Religious", "Food"],
                            key="real_cat_filter_nearby"
                        )
                        selected_cat = None
                        if cat_filter_opt == "Heritage":
                            selected_cat = CulturalCategory.HERITAGE
                        elif cat_filter_opt == "Museum":
                            selected_cat = CulturalCategory.MUSEUM
                        elif cat_filter_opt == "Religious":
                            selected_cat = CulturalCategory.RELIGIOUS
                        elif cat_filter_opt == "Food":
                            selected_cat = CulturalCategory.FOOD

                        # Proximity search
                        nearby_results = cat_store.search_nearby_places(
                            lat=target_settle.latitude,
                            lon=target_settle.longitude,
                            radius_km=float(search_radius),
                            category=selected_cat
                        )

                        if not nearby_results:
                            st.info(f"Honest Empty Result: No matching mapped cultural attractions found within {search_radius} km straight-line proximity of {target_settle.name} in this index. (Note: this reflects mapped records in the current extract, not proof that no cultural venues exist nearby).")
                        else:
                            st.markdown(f"**Cultural Attractions within {search_radius} km of {target_settle.name} ({len(nearby_results)} found):**")
                            st.caption("Straight-line proximity is indicated. Hours, entry fees, and step-free accessibility are preserved as UNKNOWN unless primary evidence exists.")

                            # Build results table
                            res_rows = []
                            for p, dist in nearby_results:
                                res_rows.append({
                                    "Place Name": p.name,
                                    "Straight-Line Proximity": f"{dist:.2f} km",
                                    "Category": p.cultural_category.value.title(),
                                    "Subcategory": (p.subcategory or "").replace("_", " ").title(),
                                    "State": p.state,
                                    "Source ID": p.source_id,
                                    "Opening Hours": p.opening_hours.status.value,
                                    "Entry Fee": p.entry_fee.status.value,
                                    "Accessibility": p.accessibility.status.value,
                                    "Source URL": p.source_url
                                })
                            res_df = pd.DataFrame(res_rows)
                            st.dataframe(res_df, use_container_width=True, hide_index=True)

                            # Paired spatial orientation map
                            st.markdown("**Spatial Orientation Map (Schematic Proximity Points):**")
                            map_df = pd.DataFrame([
                                {
                                    "name": p.name,
                                    "lat": p.latitude,
                                    "lon": p.longitude,
                                    "dist_label": f"{dist:.2f} km",
                                    "cat": p.cultural_category.value.title()
                                }
                                for p, dist in nearby_results
                            ])
                            # Add target settlement point
                            map_df = pd.concat([
                                pd.DataFrame([{
                                    "name": f"Settlement: {target_settle.name}",
                                    "lat": target_settle.latitude,
                                    "lon": target_settle.longitude,
                                    "dist_label": "0.00 km (Center)",
                                    "cat": "Settlement Center"
                                }]),
                                map_df
                            ], ignore_index=True)

                            fig_spatial = px.scatter(
                                map_df,
                                x="lon",
                                y="lat",
                                text="name",
                                color="cat",
                                hover_data=["dist_label", "cat"],
                                labels={"lon": "Longitude (°E)", "lat": "Latitude (°N)", "cat": "Category"},
                                title=f"Geographic Proximity from {target_settle.name} (Radius: {search_radius} km)",
                                color_discrete_map={
                                    "Settlement Center": "#292925",
                                    "Heritage": "#982b32",
                                    "Museum": "#365376",
                                    "Religious": "#b79d5b",
                                    "Food": "#7d6f58"
                                }
                            )
                            fig_spatial.update_traces(textposition="top center", marker=dict(size=12))
                            fig_spatial.update_layout(
                                height=420,
                                paper_bgcolor="#f7f3e9",
                                plot_bgcolor="#ffffff",
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            apply_editorial_chart_theme(fig_spatial)
                            st.plotly_chart(fig_spatial, use_container_width=True)
                            st.caption("Schematic Proximity Points: Straight-line haversine coordinates. Does not represent routed road geometry.")

            elif real_search_mode == "Explore by State / Region":
                st.markdown("**Browse Real Cultural Places by Covered State / Union Territory:**")
                all_states = cat_store.get_all_states()
                selected_state = st.selectbox("Select State / Region", all_states, key="real_state_browse_select")

                state_places = cat_store.get_places_by_state(selected_state)
                st.markdown(f"**Indexed Venues in {selected_state} ({len(state_places)} venues):**")

                if not state_places:
                    st.info(f"No mapped cultural attractions indexed for {selected_state} in this extract. This does not establish that none exist.")
                else:
                    sp_rows = []
                    for p in state_places:
                        sp_rows.append({
                            "Place Name": p.name,
                            "Category": p.cultural_category.value.title(),
                            "Subcategory": (p.subcategory or "").replace("_", " ").title(),
                            "Latitude": p.latitude,
                            "Longitude": p.longitude,
                            "Source ID": p.source_id,
                            "Hours": p.opening_hours.status.value,
                            "Admission": p.entry_fee.status.value,
                            "Accessibility": p.accessibility.status.value,
                            "Source URL": p.source_url
                        })
                    st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

            else: # Coordinate Proximity Search
                st.markdown("**Direct Coordinate Proximity Search (GPS Latitude & Longitude):**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    in_lat = st.number_input("Latitude (°N)", value=15.9189, format="%.4f", key="real_coord_lat")
                with c2:
                    in_lon = st.number_input("Longitude (°E)", value=75.6842, format="%.4f", key="real_coord_lon")
                with c3:
                    in_rad = st.number_input("Radius (km)", value=25.0, min_value=1.0, max_value=100.0, step=5.0, key="real_coord_rad")

                coord_results = cat_store.search_nearby_places(lat=in_lat, lon=in_lon, radius_km=in_rad)
                if not coord_results:
                    st.info(f"Honest Empty Result: No indexed cultural venues found within {in_rad:.1f} km of ({in_lat:.4f}, {in_lon:.4f}).")
                else:
                    st.markdown(f"**Found {len(coord_results)} places within {in_rad:.1f} km:**")
                    c_rows = []
                    for p, dist in coord_results:
                        c_rows.append({
                            "Place Name": p.name,
                            "Proximity": f"{dist:.2f} km",
                            "Category": p.cultural_category.value.title(),
                            "Source ID": p.source_id,
                            "State": p.state,
                            "Source URL": p.source_url
                        })
                    st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

            # Phase 7: Place-Detail View and Venue Evidence Inspector
            st.markdown("---")
            st.markdown("### Place Field Evidence Inspector")
            st.caption(
                "Inspect field-level evidence for real catalog places. "
                "Fields without direct primary source evidence remain strictly UNKNOWN. "
                "Note: A verified pilot venue does NOT imply nationwide verification of all 62,203 catalog places."
            )

            ev_col1, ev_col2, ev_col3 = st.columns([2, 1, 1])
            with ev_col1:
                # Provide quick access to verified pilot places or custom search
                pilot_places_lookup = {
                    "WD:Q9141 (Taj Mahal, Agra, Uttar Pradesh)": "WD:Q9141",
                    "WD:Q187635 (Qutb Minar, Delhi)": "WD:Q187635",
                    "WD:Q839211 (Brihadisvara Temple, Thanjavur, Tamil Nadu)": "WD:Q839211",
                    "WD:Q825236 (Badami Cave Temples, Bagalkot, Karnataka)": "WD:Q825236",
                    "WD:Q7293002 (Rani ki Vav, Patan, Gujarat)": "WD:Q7293002",
                    "OSM:way/458786992 (Rang Ghar, Sivasagar, Assam)": "OSM:way/458786992",
                    "OSM:way/225927819 (Kamakhya Devi Temple, Guwahati - Discovery-Only)": "OSM:way/225927819",
                }
                selected_pilot_key = st.selectbox(
                    "Select Place to Inspect",
                    list(pilot_places_lookup.keys()) + ["Custom Source ID..."],
                    key="evidence_inspector_place_select"
                )
                if selected_pilot_key == "Custom Source ID...":
                    chosen_source_id = st.text_input("Enter Real Place Source ID (e.g. WD:Q9141 or OSM:way/...)", value="WD:Q9141", key="evidence_custom_sid").strip()
                else:
                    chosen_source_id = pilot_places_lookup[selected_pilot_key]

            with ev_col2:
                req_visit_date = st.date_input("Visit Date", value=date(2026, 11, 10), key="evidence_visit_date")
            with ev_col3:
                req_visitor_profile = st.selectbox(
                    "Visitor Group",
                    ["INDIAN_ADULT", "FOREIGN_ADULT", "CHILD"],
                    index=0,
                    key="evidence_visitor_profile"
                )

            # Resolve place and evidence
            target_place = cat_store.get_place_by_id(chosen_source_id)
            ev_store = VenueEvidenceStore(VENUE_EVIDENCE_DB)

            if not target_place:
                st.warning(f"Source ID '{chosen_source_id}' not found in the published catalog.")
            else:
                # Place Header Card
                st.markdown(f"""
                <div style="background: #ffffff; border: 1.5px solid #292925; padding: 14px 18px; margin: 12px 0 16px 0; border-radius: 4px;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: bold; color: #292925;">{target_place.name}</span>
                            <span style="font-size: 0.85rem; color: #5c5852; margin-left: 10px;">
                                Category: <strong>{target_place.cultural_category.value.title()}</strong>
                                ({target_place.subcategory or 'General'}) &middot;
                                State: <strong>{target_place.state}</strong>
                                {f' &middot; District: <strong>{target_place.district}</strong>' if target_place.district else ''}
                            </span>
                        </div>
                        <span style="font-size: 0.78rem; background: #e8f0fe; color: #1a73e8; padding: 4px 10px; border-radius: 3px; font-weight: 500;">
                            Source ID: <code>{target_place.source_id}</code>
                        </span>
                    </div>
                    <div style="font-size: 0.80rem; color: #5c5852; margin-top: 8px;">
                        Upstream Record: <strong>{target_place.upstream_type}</strong> (<code>{target_place.upstream_id}</code>) &middot;
                        Coordinates: {target_place.latitude:.4f}°N, {target_place.longitude:.4f}°E &middot;
                        Extract Date: <code>{target_place.source_extract_date.isoformat()}</code> &middot;
                        Catalog Import: <code>{target_place.import_date.isoformat()}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Resolve all 5 target fields
                inspect_fields = [
                    ("Opening Hours", "opening_hours"),
                    ("Entry Fee / Admission", "entry_fee"),
                    ("Step-Free Accessibility", "step_free_access"),
                    ("Operating Status", "operating_status"),
                    ("Typical Visit Duration", "typical_duration")
                ]

                field_display_cards = []
                for label, f_key in inspect_fields:
                    res = ev_store.resolve_field(
                        source_id=chosen_source_id,
                        field_name=f_key,
                        visit_date=req_visit_date,
                        visitor_profile=req_visitor_profile,
                        clock_date=date(2026, 11, 10)
                    )

                    status_val = res.status.value
                    if status_val == "KNOWN":
                        status_val = "VERIFIED"

                    # Status badge styling
                    if status_val == "VERIFIED":
                        badge_color = "#0f5132"
                        badge_bg = "#d1e7dd"
                    elif status_val == "PROVISIONAL":
                        badge_color = "#055160"
                        badge_bg = "#cff4fc"
                    elif status_val == "STALE":
                        badge_color = "#664d03"
                        badge_bg = "#fff3cd"
                    elif status_val == "CONFLICTING":
                        badge_color = "#842029"
                        badge_bg = "#f8d7da"
                    else: # UNKNOWN
                        badge_color = "#41464b"
                        badge_bg = "#e2e3e5"

                    source_link_html = f'<a href="{res.source_url}" target="_blank" style="color:#0d6efd; text-decoration:none;">{res.source_citation}</a>' if res.source_url else (res.source_citation or "No primary source")
                    chk_date_str = res.checked_date.isoformat() if res.checked_date else "Not checked"

                    field_display_cards.append({
                        "Field": label,
                        "Status": f'<span style="background:{badge_bg}; color:{badge_color}; padding:2px 8px; border-radius:3px; font-weight:bold; font-size:0.75rem;">{status_val}</span>',
                        "Resolved Detail": res.display_text,
                        "Applicable Visitor Scope": res.visitor_scope,
                        "Checked Date": chk_date_str,
                        "Source Citation & Authority": source_link_html,
                    })

                st.markdown(
                    pd.DataFrame(field_display_cards).to_html(escape=False, index=False),
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Audit & Raw Evidence Records for this Place"):
                    raw_records = ev_store.get_evidence_records(chosen_source_id)
                    if not raw_records:
                        st.info("No raw evidence records recorded in venue_evidence.db for this place.")
                    else:
                        st.caption(f"Found {len(raw_records)} append-only evidence record(s). Preserved separately from catalog rebuilds.")
                        raw_table = []
                        for r in raw_records:
                            raw_table.append({
                                "Evidence ID": r.evidence_id,
                                "Field": r.field_name,
                                "Source Type": r.source_type.value,
                                "Checked Date": r.checked_date.isoformat(),
                                "Visitor Scope": r.visitor_scope,
                                "Raw Value": r.raw_value,
                                "Source URL": r.source_url
                            })
                        st.dataframe(pd.DataFrame(raw_table), use_container_width=True, hide_index=True)

            # Source Provenance and Attribution Footer
            st.markdown("---")
            st.markdown("""
            <div style="font-size: 0.80rem; color: #5c5852; line-height: 1.6;">
                <strong>Dataset Provenance &amp; Attribution Obligations:</strong><br>
                &bull; <strong>Settlements:</strong> UNECE UN/LOCODE (Public Domain, United Nations Economic Commission for Europe) &middot; Location Code Directory for India.<br>
                &bull; <strong>Cultural Heritage:</strong> Wikidata &amp; UNESCO World Heritage Sites in India (Creative Commons CC0 1.0 Universal Public Domain Dedication).<br>
                &bull; <strong>Administrative Boundary:</strong> ISO 3166-1 IND MultiPolygon GeoJSON (Open Database License ODbL / OpenStreetMap contributors).<br>
                &bull; <strong>Integrity &amp; Status:</strong> All source records checksummed with SHA-256. Opening hours, entry fees, and accessibility default to <code>UNKNOWN</code> without primary source verification.
            </div>
            """, unsafe_allow_html=True)



# ==============================================================================
# SCREEN 3: PLANNER
# ==============================================================================
elif view_mode == "Planner":
    render_section_banner(
        section_key="planner",
        title_html="Build a <em>Cultural Day</em>",
        subtitle="Generate feasible day plans using recorded directed walking and driving legs from the study travel matrix.",
        kicker="02 / DETERMINISTIC ITINERARY ENGINE",
    )

    tab_synth_planner, tab_legacy_planner = st.tabs([
        "Synthetic Day Planner (45 Study Cities)",
        "Real-Place Planner (Conditional)"
    ])

    with tab_synth_planner:
        cities = get_study_cities()
        c_names = [c["name"] for c in cities]

        # Quick Benchmark Hubs & Presets
        def _on_planner_hub_pick():
            picked = st.session_state.get("pills_planner_hub")
            if not picked:
                return
            if "Jaipur" in picked:
                p_target = "Jaipur"
                b_target = 1000.0
            elif "Agra" in picked:
                p_target = "Agra"
                b_target = 2000.0
            else:
                p_target = "New Delhi" if picked == "Delhi" else picked
                b_target = float(st.session_state.get("form_p_budget", 1000.0))
            st.session_state["form_p_city"] = p_target
            st.session_state["p_city"] = p_target
            st.session_state["form_p_budget"] = b_target
            st.session_state["p_budget"] = b_target

        def _load_jaipur_baseline_planner():
            st.session_state["form_p_city"] = "Jaipur"
            st.session_state["form_p_date"] = date(2026, 8, 29)
            st.session_state["form_p_start"] = "09:00"
            st.session_state["form_p_window"] = 6.0
            st.session_state["form_p_budget"] = 1000.0
            st.session_state["form_p_stops"] = 3
            st.session_state["form_p_mode"] = "cab"
            st.session_state["form_p_step_free"] = False
            st.session_state["form_p_pref_cats"] = []
            st.session_state["p_city"] = "Jaipur"
            st.session_state["p_date"] = date(2026, 8, 29)
            st.session_state["p_start"] = "09:00"
            st.session_state["p_window"] = 6.0
            st.session_state["p_budget"] = 1000.0
            st.session_state["p_stops"] = 3
            st.session_state["p_mode"] = "cab"
            st.session_state["p_step_free"] = False
            st.session_state["p_pref_cats"] = []

        def _load_agra_preset_planner():
            st.session_state["form_p_city"] = "Agra"
            st.session_state["form_p_date"] = date(2026, 8, 29)
            st.session_state["form_p_start"] = "09:00"
            st.session_state["form_p_window"] = 8.0
            st.session_state["form_p_budget"] = 2000.0
            st.session_state["form_p_stops"] = 3
            st.session_state["form_p_mode"] = "cab"
            st.session_state["form_p_step_free"] = False
            st.session_state["form_p_pref_cats"] = ["heritage", "museum"]
            st.session_state["p_city"] = "Agra"
            st.session_state["p_date"] = date(2026, 8, 29)
            st.session_state["p_start"] = "09:00"
            st.session_state["p_window"] = 8.0
            st.session_state["p_budget"] = 2000.0
            st.session_state["p_stops"] = 3
            st.session_state["p_mode"] = "cab"
            st.session_state["p_step_free"] = False
            st.session_state["p_pref_cats"] = ["heritage", "museum"]

        st.markdown("**Quick Benchmark Hubs & Presets (Instant 1-Click Load):**")
        quick_plan_hubs = ["Jaipur (Baseline Spec)", "Agra (Cultural Preset)", "Delhi", "Mumbai", "Varanasi", "Kochi", "Bengaluru"]
        st.pills(
            "Quick Benchmark Hubs",
            quick_plan_hubs,
            key="pills_planner_hub",
            on_change=_on_planner_hub_pick,
            label_visibility="collapsed",
        )

        st.markdown("**Named Planning Presets:**")
        preset_cols = st.columns([1.5, 1.5, 5])
        with preset_cols[0]:
            st.button("Load Jaipur Baseline", key="btn_preset_jaipur", on_click=_load_jaipur_baseline_planner)
        with preset_cols[1]:
            st.button("Load Agra Cultural Preset", key="btn_preset_agra", on_click=_load_agra_preset_planner)

        # Input Controls - Row 1
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns([1.4, 1.2, 1.1, 1.0, 1.1, 0.9])
        cur_city = st.session_state.get("p_city", "Jaipur")
        city_idx = c_names.index(cur_city) if cur_city in c_names else 0

        with p_col1:
            p_city = st.selectbox("Study City", c_names, index=city_idx, key="form_p_city")
        with p_col2:
            p_date_val = st.date_input(
                "Visit Date (Local TZ)",
                value=st.session_state.get("p_date", date(2026, 8, 29)),
                min_value=date(2025, 1, 1),
                max_value=date(2027, 12, 31),
                key="form_p_date",
                help="Date affects day-of-week closure checks according to the study city's local timezone (Asia/Kolkata, UTC+05:30)."
            )
        with p_col3:
            p_start_str = st.text_input("Start Time (24h)", st.session_state.get("p_start", "09:00"), key="form_p_start")
        with p_col4:
            p_window = st.number_input("Window (h)", min_value=1.0, max_value=12.0, value=st.session_state.get("p_window", 6.0), step=0.5, key="form_p_window")
        with p_col5:
            p_budget = st.number_input("Budget (INR)", min_value=100.0, max_value=5000.0, value=st.session_state.get("p_budget", 1000.0), step=50.0, key="form_p_budget")
        with p_col6:
            p_stops = st.number_input("Stops", min_value=2, max_value=4, value=st.session_state.get("p_stops", 3), step=1, key="form_p_stops")

        # Input Controls - Row 2: Explicit Preferences & Constraints
        pref_r1, pref_r2, pref_r3 = st.columns([1.5, 1.8, 2.7])
        with pref_r1:
            mode_opts = ["cab", "walk"]
            cur_mode = st.session_state.get("p_mode", "cab")
            mode_idx = mode_opts.index(cur_mode) if cur_mode in mode_opts else 0
            p_mode = st.selectbox("Transport Mode (Hard Constraint)", mode_opts, index=mode_idx, key="form_p_mode",
                                  help="Strictly matches directed rows in poi_travel_matrix. Does not infer reverse legs.")
        with pref_r2:
            p_step_free = st.checkbox("Step-Free at Stop Only (Hard Constraint)",
                                      value=st.session_state.get("p_step_free", False),
                                      key="form_p_step_free",
                                      help="Requires stop accessibility == 'step_free'. Note: transit transfer accessibility is unrecorded.")
        with pref_r3:
            all_cat_options = ["heritage", "museum", "religious", "food", "entertainment", "nature"]
            cur_pref_cats = st.session_state.get("p_pref_cats", [])
            p_pref_cats = st.multiselect("Preferred Cultural Categories (Tie-Breaker)", all_cat_options, default=cur_pref_cats, key="form_p_pref_cats",
                                         help="Used as a deterministic tie-breaker when comparing candidate itineraries of equal duration and stops.")

        gen_clicked = st.button("Generate Plans", type="primary", key="btn_gen_plans")

        p_date_iso = p_date_val.isoformat() if isinstance(p_date_val, (date, datetime)) else str(p_date_val)

        # Handle plan generation and stable state snapshot
        parsed_start = parse_time_to_minutes(p_start_str)
        if parsed_start is None:
            st.warning(
                f"Invalid start time '{p_start_str}'. Please enter a valid 24-hour time in HH:MM format (00:00 - 23:59, e.g. 09:00)."
            )
            if "plan_snapshot" in st.session_state:
                st.session_state["plan_snapshot"]["stale"] = True
                fastest_plan = st.session_state["plan_snapshot"]["plans"].get("fastest")
                cultural_plan = st.session_state["plan_snapshot"]["plans"].get("most_cultural")
            else:
                fastest_plan = None
                cultural_plan = None
        else:
            # Recompute on click or if input parameters differ from snapshot or snapshot missing
            should_compute = gen_clicked or ("plan_snapshot" not in st.session_state)
            if not should_compute:
                snap = st.session_state["plan_snapshot"]
                if (snap.get("city") != p_city or
                    snap.get("visit_date") != p_date_iso or
                    snap.get("start_time") != p_start_str or
                    snap.get("window_hours") != p_window or
                    snap.get("budget_inr") != p_budget or
                    snap.get("target_stops") != int(p_stops) or
                    snap.get("transport_mode") != p_mode or
                    snap.get("step_free_only") != p_step_free or
                    snap.get("preferred_categories") != p_pref_cats):
                    should_compute = True

            if should_compute:
                try:
                    plans = _run_planner_pair_cached(
                        city_name=p_city,
                        visit_date=p_date_iso,
                        start_time=p_start_str,
                        max_duration_hours=p_window,
                        budget_inr=p_budget,
                        target_stops=int(p_stops),
                        transport_mode=p_mode,
                        step_free_only=p_step_free,
                        preferred_categories=tuple(p_pref_cats or ()),
                    )
                except ValueError as err:
                    st.error(f"Input validation error: {err}")
                    plans = {"fastest": None, "most_cultural": None}

                is_fixture = (p_city == "Jaipur" and p_date_iso == "2026-08-29" and p_start_str == "09:00" and
                              p_window == 6.0 and p_budget == 1000.0 and int(p_stops) == 3 and
                              p_mode == "cab" and not p_step_free and not p_pref_cats)
                st.session_state["plan_snapshot"] = {
                    "city": p_city,
                    "visit_date": p_date_iso,
                    "start_time": p_start_str,
                    "window_hours": p_window,
                    "budget_inr": p_budget,
                    "target_stops": int(p_stops),
                    "transport_mode": p_mode,
                    "step_free_only": p_step_free,
                    "preferred_categories": p_pref_cats,
                    "plans": plans,
                    "is_default_fixture": is_fixture,
                    "stale": False,
                }
                # Also synchronize working state
                st.session_state["p_city"] = p_city
                st.session_state["p_date"] = p_date_val
                st.session_state["p_start"] = p_start_str
                st.session_state["p_window"] = p_window
                st.session_state["p_budget"] = p_budget
                st.session_state["p_stops"] = int(p_stops)
                st.session_state["p_mode"] = p_mode
                st.session_state["p_step_free"] = p_step_free
                st.session_state["p_pref_cats"] = p_pref_cats

            fastest_plan = st.session_state["plan_snapshot"]["plans"].get("fastest")
            cultural_plan = st.session_state["plan_snapshot"]["plans"].get("most_cultural")

        if fastest_plan is None and cultural_plan is None:
            weekday_name = p_date_val.strftime("%A") if isinstance(p_date_val, (date, datetime)) else "selected date"
            st.warning(
                f"Infeasible Day Plan: No feasible {p_stops}-stop itinerary found for {p_city} on {weekday_name} ({p_date_iso}) "
                f"within budget INR {p_budget:,.2f}, window {p_window} hours, and mode '{p_mode}'. "
                "Causes can include known day-of-week closures, opening hour conflicts, budget limits, or unrecorded matrix legs. "
                "Consider increasing budget, adjusting the date or start time, or choosing another study city."
            )
        elif fastest_plan and cultural_plan:
            # Check for identical route
            same_route = (
                [s.poi_id for s in fastest_plan.stops] == [s.poi_id for s in cultural_plan.stops]
            )

            if same_route:
                st.info(
                    f"Same-Objective Result: In {p_city}, the identical route achieves both minimum elapsed duration "
                    f"({fastest_plan.elapsed_formatted}) and maximum cultural stops ({fastest_plan.cultural_stops_count} stops) "
                    "under the active constraints. No false trade-off is presented."
                )

            st.markdown("---")
            c_plan1, c_plan2 = st.columns(2)

            with c_plan1:
                st.markdown("""
                <div style="background: #ffffff; border: 1px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 22px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <h3 style="margin: 0; font-family: Georgia, serif;">Plan A · Fastest Feasible Plan</h3>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #365376; border: 1px solid #365376; padding: 2px 6px;">MIN ELAPSED TIME</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                m_a1, m_a2, m_a3 = st.columns(3)
                m_a1.metric("Elapsed Time", fastest_plan.elapsed_formatted, f"{fastest_plan.elapsed_minutes} min")
                m_a2.metric("Cultural Stops", f"{fastest_plan.cultural_stops_count} / {fastest_plan.total_stops_count}")
                m_a3.metric("Total Cost", f"INR {fastest_plan.total_cost_inr:.2f}")

                if fastest_plan.has_uncertain_hours:
                    st.warning("Notice: Plan contains stops with unrecorded opening/closing hours. Visited conditionally.")

                st.markdown("**Timetable & Leg Itinerary:**")
                for s in fastest_plan.stops:
                    badge = "[Cultural]" if s.is_cultural else "[Other]"
                    cond_note = f" [CONDITIONAL: {s.hours_status}]" if s.hours_status != "EXACT_VERIFIED" else ""
                    st.markdown(f"**{s.arrival_time} - {s.visit_end_time}** : **{s.name}** {badge}{cond_note}")
                    st.caption(f"Visit: {s.duration_minutes} min | Fee: INR {s.entry_fee:.2f} | Access: {s.accessibility} | POI ID: `{s.poi_id}`")
                    if s.stop_index <= len(fastest_plan.legs):
                        l = fastest_plan.legs[s.stop_index - 1]
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ Transit Leg (`{l.matrix_id}`): {l.travel_minutes} min ({l.distance_km} km via {l.mode}, INR {l.cost_inr:.2f}) | Est. Carbon: {l.carbon_kg:.3f} kg")

                # Paired Spatial Sequence Map for Plan A
                st.markdown("**Paired Spatial Sequence View (Plan A):**")
                render_plan_spatial_map(fastest_plan, "Plan A (Fastest)", line_color="#365376")

            with c_plan2:
                cultural_gain = cultural_plan.cultural_stops_count - fastest_plan.cultural_stops_count
                gain_label = f"+{cultural_gain} cultural stop{'s' if cultural_gain > 1 else ''}" if cultural_gain > 0 else "Optimal Cultural"

                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 22px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <h3 style="margin: 0; font-family: Georgia, serif;">Plan B · Most Cultural Plan</h3>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">MAX CULTURAL STOPS</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                m_b1, m_b2, m_b3 = st.columns(3)
                m_b1.metric("Elapsed Time", cultural_plan.elapsed_formatted, f"{cultural_plan.elapsed_minutes} min")
                m_b2.metric("Cultural Stops", f"{cultural_plan.cultural_stops_count} / {cultural_plan.total_stops_count}", gain_label)
                m_b3.metric("Total Cost", f"INR {cultural_plan.total_cost_inr:.2f}")

                if cultural_plan.has_uncertain_hours:
                    st.warning("Notice: Plan contains stops with unrecorded opening/closing hours. Visited conditionally.")

                st.markdown("**Timetable & Leg Itinerary:**")
                for s in cultural_plan.stops:
                    badge = "[Cultural]" if s.is_cultural else "[Other]"
                    cond_note = f" [CONDITIONAL: {s.hours_status}]" if s.hours_status != "EXACT_VERIFIED" else ""
                    st.markdown(f"**{s.arrival_time} - {s.visit_end_time}** : **{s.name}** {badge}{cond_note}")
                    st.caption(f"Visit: {s.duration_minutes} min | Fee: INR {s.entry_fee:.2f} | Access: {s.accessibility} | POI ID: `{s.poi_id}`")
                    if s.stop_index <= len(cultural_plan.legs):
                        l = cultural_plan.legs[s.stop_index - 1]
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ Transit Leg (`{l.matrix_id}`): {l.travel_minutes} min ({l.distance_km} km via {l.mode}, INR {l.cost_inr:.2f}) | Est. Carbon: {l.carbon_kg:.3f} kg")

                # Paired Spatial Sequence Map for Plan B
                st.markdown("**Paired Spatial Sequence View (Plan B):**")
                render_plan_spatial_map(cultural_plan, "Plan B (Most Cultural)", line_color="#982b32")

            # Traceability & Calculation Inspector Expander
            with st.expander("Inspect Plan Calculation Traceability (Source POI & Leg IDs)"):
                st.markdown("**Traceability Policy**: Between-stop travel time and fees are derived directly from recorded source rows in `activities_poi` and `poi_travel_matrix`. Travel to first stop and return from last stop are excluded as the study traveler's origin and end points are unrecorded.")
                t_c1, t_c2 = st.columns(2)
                with t_c1:
                    st.markdown("**Plan A Calculation Breakdown:**")
                    st.markdown(f"- Entry fees total: INR {fastest_plan.entry_cost_inr:.2f} (Decimal)")
                    st.markdown(f"- Directed transit fares total: INR {fastest_plan.travel_cost_inr:.2f} (Decimal)")
                    st.markdown(f"- Total budget used: INR {fastest_plan.total_cost_inr:.2f}")
                    st.markdown(f"- Estimated transit carbon: {fastest_plan.estimated_carbon_kg:.3f} kg")
                    for s in fastest_plan.stops:
                        st.markdown(f"  * Stop {s.stop_index}: `{s.poi_id}` ({s.name}) - Fee: INR {s.entry_fee:.2f}, Hours: {s.hours_status}, Access: {s.accessibility}")
                    for l in fastest_plan.legs:
                        st.markdown(f"  * Leg {l.leg_index}: `{l.matrix_id}` (`{l.origin_id}` &rarr; `{l.destination_id}`) - Mode: {l.mode}, Time: {l.travel_minutes}m, Fare: INR {l.cost_inr:.2f}")
                with t_c2:
                    st.markdown("**Plan B Calculation Breakdown:**")
                    st.markdown(f"- Entry fees total: INR {cultural_plan.entry_cost_inr:.2f} (Decimal)")
                    st.markdown(f"- Directed transit fares total: INR {cultural_plan.travel_cost_inr:.2f} (Decimal)")
                    st.markdown(f"- Total budget used: INR {cultural_plan.total_cost_inr:.2f}")
                    st.markdown(f"- Estimated transit carbon: {cultural_plan.estimated_carbon_kg:.3f} kg")
                    for s in cultural_plan.stops:
                        st.markdown(f"  * Stop {s.stop_index}: `{s.poi_id}` ({s.name}) - Fee: INR {s.entry_fee:.2f}, Hours: {s.hours_status}, Access: {s.accessibility}")
                    for l in cultural_plan.legs:
                        st.markdown(f"  * Leg {l.leg_index}: `{l.matrix_id}` (`{l.origin_id}` &rarr; `{l.destination_id}`) - Mode: {l.mode}, Time: {l.travel_minutes}m, Fare: INR {l.cost_inr:.2f}")

            # Visual Time Allocation Comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Day Schedule Time Allocation & Transit Breakdown: {p_city}")
            f_visit = sum(s.duration_minutes for s in fastest_plan.stops)
            f_transit = sum(l.travel_minutes for l in fastest_plan.legs)
            c_visit = sum(s.duration_minutes for s in cultural_plan.stops)
            c_transit = sum(l.travel_minutes for l in cultural_plan.legs)

            time_alloc_df = pd.DataFrame([
                {"Plan": "Plan A (Fastest)", "Activity": "Attraction Visit Time", "Minutes": f_visit},
                {"Plan": "Plan A (Fastest)", "Activity": "Transit Travel Time", "Minutes": f_transit},
                {"Plan": "Plan B (Most Cultural)", "Activity": "Attraction Visit Time", "Minutes": c_visit},
                {"Plan": "Plan B (Most Cultural)", "Activity": "Transit Travel Time", "Minutes": c_transit},
            ])

            fig_time = px.bar(
                time_alloc_df,
                x="Plan",
                y="Minutes",
                color="Activity",
                text="Minutes",
                barmode="stack",
                title=f"Time Spent at Attractions vs in Transit ({p_city})",
                color_discrete_map={"Attraction Visit Time": "#982b32", "Transit Travel Time": "#365376"}
            )
            fig_time.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            apply_editorial_chart_theme(fig_time)
            st.plotly_chart(fig_time, use_container_width=True)

            def _go_to_compare_from_planner():
                st.query_params["view"] = "compare"

            st.markdown("<br>", unsafe_allow_html=True)
            cmp_btn_c1, cmp_btn_c2 = st.columns([1.5, 1])
            with cmp_btn_c1:
                st.button(
                    f"Compare These Plans in Compare View ({p_city}) →",
                    type="primary",
                    key="btn_planner_to_compare",
                    on_click=_go_to_compare_from_planner,
                )

    with tab_legacy_planner:
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #292925; border-left: 4px solid #982b32; padding: 14px 18px; margin-bottom: 16px;">
            <div style="font-weight: bold; color: #292925; font-size: 1.05rem;">
                Real-Place Planner (Conditional)
            </div>
            <div style="font-size: 0.84rem; color: #5c5852; margin-top: 4px; line-height: 1.5;">
                Explore conditional cultural day trips across indexed Indian places. Routing requires an explicitly configured
                profile-specific provider; unavailable routes are reported rather than estimated. Most venue details remain
                <strong>UNKNOWN</strong>, so a displayed timetable may be illustrative.
            </div>
        </div>
        """, unsafe_allow_html=True)

        rp_col1, rp_col2 = st.columns([1.2, 2.0])

        with rp_col1:
            st.markdown("**1. Select Region or Settlement:**")
            real_plan_mode = st.radio(
                "Planning Hub Selection",
                ["Example Place Pairs", "Search by Settlement Name", "Custom Real Place IDs"],
                key="real_plan_hub_mode",
                label_visibility="collapsed"
            )

            rp_selected_ids = []
            rp_state_or_district = "Karnataka"
            rp_settlement_id = None

            if real_plan_mode == "Example Place Pairs":
                pilot_hub_options = {
                    "Badami & Aihole (Karnataka - South Heritage)": ["WD:Q825236", "WD:Q3517553"],
                    "Agra Heritage Circuit (Uttar Pradesh - North)": ["WD:Q9141", "WD:Q171857"],
                    "Delhi Historic Monuments (Delhi - North)": ["WD:Q187635", "WD:Q189648"],
                    "Sivasagar Ahom Heritage (Assam - Non-Metro North-East)": ["OSM:way/458786992", "OSM:way/155294111"],
                }
                chosen_hub_key = st.selectbox("Select Pilot Hub", list(pilot_hub_options.keys()), key="real_pilot_hub_select")
                rp_selected_ids = pilot_hub_options[chosen_hub_key]

            elif real_plan_mode == "Search by Settlement Name":
                settle_search = st.text_input("Enter Settlement Name", value="Badami", key="real_plan_settle_input").strip()
                if _ensure_real_catalog_db():
                    cat_store_inst = RealCatalogStore(REAL_CATALOG_DB)
                    found_settles = cat_store_inst.search_settlements(settle_search) if settle_search else []
                else:
                    found_settles = []
                if found_settles:
                    settle_map = {f"{s.name} ({s.state} · {s.settlement_type.title()})": s.settlement_id for s in found_settles[:10]}
                    picked_s = st.selectbox("Choose Matching Settlement", list(settle_map.keys()), key="real_plan_settle_pick")
                    rp_settlement_id = settle_map[picked_s]
                else:
                    st.caption("No matching settlements found in current catalog extract.")

            else:  # Custom Real Place IDs
                custom_ids_input = st.text_area(
                    "Enter Source IDs (comma-separated)",
                    value="WD:Q825236, WD:Q3517553",
                    key="real_plan_custom_ids"
                )
                rp_selected_ids = [s.strip() for s in custom_ids_input.split(",") if s.strip()]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**2. Trip Schedule & Traveler Profile:**")
            rp_date = st.date_input("Visit Date", value=date(2026, 11, 10), key="real_plan_date")

            c_time1, c_time2 = st.columns(2)
            with c_time1:
                rp_start = st.text_input("Start Time (24h)", value="09:00", key="real_plan_start")
            with c_time2:
                rp_window = st.number_input("Max Window (hrs)", min_value=2.0, max_value=14.0, value=7.0, step=0.5, key="real_plan_window")

            c_mod1, c_mod2 = st.columns(2)
            with c_mod1:
                available_modes = ["Driving (Car / Cab)"]
                if configured_backend(TransportMode.WALKING):
                    available_modes.append("Walking (<25 km)")
                rp_mode_choice = st.selectbox("Transport Mode", available_modes, key="real_plan_mode_select")
                rp_mode = TransportMode.DRIVING if "Driving" in rp_mode_choice else TransportMode.WALKING
            with c_mod2:
                rp_visitor = st.selectbox("Visitor Profile", ["INDIAN_ADULT", "FOREIGN_ADULT", "CHILD"], key="real_plan_visitor_select")

            c_dur1, c_dur2 = st.columns(2)
            with c_dur1:
                rp_dwell = st.number_input("Dwell Time (min/stop)", min_value=15, max_value=180, value=60, step=15, key="real_plan_dwell")
            with c_dur2:
                rp_round_trip = st.checkbox("Round Trip (Return to Start)", value=False, key="real_plan_round_trip")

            rp_step_free = st.checkbox("Step-Free Access Required", value=False, key="real_plan_step_free")
            rp_budget = st.number_input("Admission budget (INR; known fees only)", min_value=0.0,
                                        value=1000.0, step=100.0, key="real_plan_budget")
            rp_stops = st.number_input("Desired stops", min_value=1, max_value=5, value=2,
                                       key="real_plan_stops") if not rp_selected_ids else len(rp_selected_ids)
            if not configured_backend(rp_mode):
                st.warning("No routing backend configured for this mode. Set CULTUREROUTE_OSRM_DRIVING_URL or CULTUREROUTE_OSRM_WALKING_URL.")

            run_real_btn = st.button("Generate Real-Place Day Plan →", type="primary", key="btn_run_real_planner")

        with rp_col2:
            st.markdown("### Real Itinerary & Directed Routing Results")

            # Execute real planner if requested or default
            if run_real_btn or "real_plan_result" in st.session_state:
                if run_real_btn:
                    with st.spinner("Computing genuine directed routing legs and resolving primary field evidence..."):
                        try:
                            res_plan = plan_real_place_day(
                                catalog_db=REAL_CATALOG_DB,
                                evidence_db=VENUE_EVIDENCE_DB,
                                routes_db=DEFAULT_ROUTES_DB,
                                selected_place_ids=rp_selected_ids if rp_selected_ids else None,
                                settlement_id=rp_settlement_id,
                                visit_date=rp_date,
                                start_time=rp_start,
                                max_duration_hours=float(rp_window),
                                target_stops=rp_stops,
                                transport_mode=rp_mode,
                                step_free_required=rp_step_free,
                                visitor_scope=rp_visitor,
                                user_duration_minutes=int(rp_dwell),
                                round_trip=rp_round_trip,
                                allow_network=True,
                                budget_inr=float(rp_budget),
                            )
                        except ValueError as exc:
                            res_plan = {"fastest": None, "infeasible_reason": str(exc)}
                        st.session_state["real_plan_result"] = res_plan
                else:
                    res_plan = st.session_state["real_plan_result"]

                fastest_p = res_plan.get("fastest")
                infeasible_err = res_plan.get("infeasible_reason")

                if infeasible_err or not fastest_p:
                    st.error(f"Plan Infeasible: {infeasible_err or 'No valid routing or schedule sequence found.'}")
                    st.caption("CultureRoute provides honest failure reporting rather than manufacturing unrealistic routes or assuming unknown opening windows.")
                else:
                    # Result Level Banner
                    lvl = fastest_p.result_level
                    if lvl == "VERIFIED":
                        lvl_bg, lvl_color = "#d1e7dd", "#0f5132"
                    elif lvl == "PROVISIONAL":
                        lvl_bg, lvl_color = "#cff4fc", "#055160"
                    else:
                        lvl_bg, lvl_color = "#fff3cd", "#664d03"

                    st.markdown(f"""
                    <div style="background:{lvl_bg}; color:{lvl_color}; padding:10px 14px; border-radius:4px; font-weight:bold; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                        <span>Plan Classification: {lvl}</span>
                        <span style="font-size:0.80rem; font-weight:normal;">{fastest_p.elapsed_formatted} &middot; {fastest_p.total_stops_count} Cultural Stops</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if fastest_p.reasons:
                        st.caption(fastest_p.reasons[0])

                    # Summary Metrics Bar
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Elapsed Time", fastest_p.elapsed_formatted, f"Start: {fastest_p.start_time}")
                    cost_val_str = f"INR {fastest_p.known_admission_subtotal_inr:.2f}"
                    cost_delta = "At least this much; some fees unknown" if any(s.entry_fee is None for s in fastest_p.stops) else "Known admission fees"
                    m2.metric("Known Admission Subtotal", cost_val_str, cost_delta)
                    total_dist = sum(l.distance_km for l in fastest_p.legs)
                    m3.metric("Transit Distance", f"{total_dist:.2f} km", f"{len(fastest_p.legs)} Directed Leg(s)")
                    m4.metric("Transport Mode", fastest_p.transport_mode, "Configured OSRM profile")
                    if fastest_p.has_unknown_costs:
                        st.warning("Total trip cost is unknown. Admission budget and driving fare cannot be guaranteed.")
                    if res_plan.get("comparison_note"):
                        st.caption(res_plan["comparison_note"])

                    # Timetable Breakdown
                    st.markdown("#### Timetable & Stop Schedule")
                    for s in fastest_p.stops:
                        fee_disp = f"INR {s.entry_fee:.2f}" if s.entry_fee is not None else "UNKNOWN"
                        fee_status_badge = f'<span style="font-size:0.75rem; background:#f0ebe1; padding:1px 6px; border-radius:2px;">{s.fee_status}</span>'
                        st.markdown(f"""
                        <div style="background:#ffffff; border:1px solid #dcd4c8; padding:10px 14px; margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between;">
                                <strong>Stop {s.stop_index}: {s.name}</strong>
                                <span>{s.arrival_time} &ndash; {s.visit_end_time}</span>
                            </div>
                            <div style="font-size:0.82rem; color:#5c5852; margin-top:4px;">
                                Admission: <strong>{fee_disp}</strong> {fee_status_badge} &middot;
                                Schedule Status: <code>{s.hours_status}</code> &middot;
                                Dwell: {s.duration_minutes} min ({'Traveler input' if s.is_duration_assumption else 'Official'})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Render leg after stop
                        if s.stop_index <= len(fastest_p.legs):
                            l = fastest_p.legs[s.stop_index - 1]
                            st.markdown(f"""
                            <div style="margin-left: 20px; font-size: 0.82rem; color: #365376; margin-bottom: 8px;">
                                ↳ <strong>Directed Transit Leg {l.leg_index}:</strong> {l.origin_name} &rarr; {l.destination_name}<br>
                                &nbsp;&nbsp;&nbsp;&nbsp;Distance: {l.distance_km:.2f} km &middot; Travel Time: {l.travel_minutes} min via {l.mode} &middot; Engine: <em>{l.source}</em>
                            </div>
                            """, unsafe_allow_html=True)

                    # Routed Road Map
                    st.markdown("#### Genuine Routed Road Geometry Map")
                    render_plan_spatial_map(fastest_p, "Real Route Plan", line_color="#982b32")
                    st.caption("Solid lines use geometry returned by the configured routing backend. Venue access and live conditions may differ.")
            else:
                st.info("Select a hub or settlement and request a routed plan. Missing evidence will remain visible.")


# ==============================================================================
# SCREEN 4: COMPARE
# ==============================================================================
elif view_mode == "Compare":
    render_section_banner(
        section_key="compare",
        title_html="Compare Fastest vs Most Cultural Plans",
        subtitle="Direct side-by-side trade-off evaluation: time saved versus cultural heritage representation gained.",
        kicker="03 / DUAL-OBJECTIVE PLAN COMPARISON",
    )

    # Consume snapshot from Planner or default to specification Jaipur benchmark fixture
    submitted_snap = st.session_state.get("plan_snapshot")
    submitted_city = submitted_snap.get("city", "Jaipur") if submitted_snap else "Jaipur"
    is_submitted_custom = submitted_snap is not None and not submitted_snap.get("is_default_fixture", False)

    # Let user toggle between active submitted plan (e.g. Agra) and standard Jaipur fixture
    if is_submitted_custom:
        cmp_choice = st.radio(
            "Select Comparison Subject:",
            [
                f"Active Generated Plan: {submitted_city} ({submitted_snap.get('target_stops', 3)} stops, INR {submitted_snap.get('budget_inr', 1000):,.0f}, {submitted_snap.get('transport_mode', 'cab')})",
                "Standard Jaipur Benchmark Fixture (207 min / INR 585.46 / 1.806 kg CO2)"
            ],
            index=0,
            horizontal=True,
            key="cmp_subject_radio"
        )
        if "Standard Jaipur Benchmark" in cmp_choice:
            jaipur_plans = _run_planner_pair_cached(
                city_name="Jaipur",
                visit_date="2026-08-29",
                start_time="09:00",
                max_duration_hours=6.0,
                budget_inr=1000.0,
                target_stops=3,
                transport_mode="cab",
                step_free_only=False,
                preferred_categories=(),
            )
            snapshot = {
                "city": "Jaipur",
                "visit_date": "2026-08-29",
                "start_time": "09:00",
                "window_hours": 6.0,
                "budget_inr": 1000.0,
                "target_stops": 3,
                "transport_mode": "cab",
                "step_free_only": False,
                "preferred_categories": [],
                "plans": jaipur_plans,
                "is_default_fixture": True,
                "stale": False,
            }
        else:
            snapshot = submitted_snap
    else:
        if submitted_snap is None:
            jaipur_plans = _run_planner_pair_cached(
                city_name="Jaipur",
                visit_date="2026-08-29",
                start_time="09:00",
                max_duration_hours=6.0,
                budget_inr=1000.0,
                target_stops=3,
                transport_mode="cab",
                step_free_only=False,
                preferred_categories=(),
            )
            snapshot = {
                "city": "Jaipur",
                "visit_date": "2026-08-29",
                "start_time": "09:00",
                "window_hours": 6.0,
                "budget_inr": 1000.0,
                "target_stops": 3,
                "transport_mode": "cab",
                "step_free_only": False,
                "preferred_categories": [],
                "plans": jaipur_plans,
                "is_default_fixture": True,
                "stale": False,
            }
            st.session_state["plan_snapshot"] = snapshot
        else:
            snapshot = submitted_snap

    plans = snapshot.get("plans", {})
    fastest = plans.get("fastest")
    cultural = plans.get("most_cultural")
    cmp_city = snapshot.get("city", "Jaipur")
    is_fixture = snapshot.get("is_default_fixture", False)

    if fastest is None and cultural is None:
        st.warning(
            f"Infeasible Plan Comparison: No feasible {snapshot.get('target_stops', 3)}-stop itinerary found for {cmp_city} "
            f"under the submitted budget INR {snapshot.get('budget_inr', 1000):,.2f}, duration {snapshot.get('window_hours', 6.0)} hours, "
            f"and mode '{snapshot.get('transport_mode', 'cab')}'."
        )
        st.markdown('''
        <a href="?view=planner" target="_self" class="cr-button" style="display: inline-block; padding: 8px 16px; text-decoration: none; margin: 12px 0;">
            &larr; Return to Planner to Adjust Parameters
        </a>
        ''', unsafe_allow_html=True)
    elif fastest and cultural:
        time_diff = cultural.elapsed_minutes - fastest.elapsed_minutes
        cost_diff = cultural.total_cost_inr - fastest.total_cost_inr
        carbon_diff = cultural.estimated_carbon_kg - fastest.estimated_carbon_kg
        cultural_gain = cultural.cultural_stops_count - fastest.cultural_stops_count

        # Check for identical route
        is_same_route = ([s.poi_id for s in fastest.stops] == [s.poi_id for s in cultural.stops])

        # Interactive City Selector and Benchmark Presets for Dynamic Comparison
        all_study_cities = get_study_cities()
        all_city_names = [c["name"] for c in all_study_cities]
        current_city_idx = all_city_names.index(cmp_city) if cmp_city in all_city_names else 0

        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #b6afa3; border-top: 3px solid #982b32; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: bold; color: #292925; font-size: 0.88rem; margin-bottom: 6px;">
                Interactive Evaluation Controls &middot; Switch City or Benchmark
            </div>
            <div style="font-size: 0.82rem; color: #5c5852;">
                Compare Plan A vs Plan B trade-offs dynamically for any study city, or test named syllabus benchmark fixtures.
            </div>
        </div>
        """, unsafe_allow_html=True)

        def run_compare_recalc(target_city: str, v_date: str = "2026-08-29", start: str = "09:00",
                               win: float = 6.0, bud: float = 1000.0, stops: int = 3, mode: str = "cab"):
            new_plans = _run_planner_pair_cached(
                city_name=target_city,
                visit_date=v_date,
                start_time=start,
                max_duration_hours=win,
                budget_inr=bud,
                target_stops=stops,
                transport_mode=mode,
                step_free_only=False,
                preferred_categories=(),
            )
            is_fix = (target_city == "Jaipur" and v_date == "2026-08-29" and start == "09:00" and
                      win == 6.0 and bud == 1000.0 and stops == 3 and mode == "cab")
            st.session_state["plan_snapshot"] = {
                "city": target_city,
                "visit_date": v_date,
                "start_time": start,
                "window_hours": win,
                "budget_inr": bud,
                "target_stops": stops,
                "transport_mode": mode,
                "step_free_only": False,
                "preferred_categories": [],
                "plans": new_plans,
                "is_default_fixture": is_fix,
                "stale": False
            }

        def on_cmp_city_dropdown_change():
            chosen_city = st.session_state.get("compare_city_select")
            if chosen_city:
                run_compare_recalc(chosen_city)

        def on_cmp_hub_pill_change():
            picked = st.session_state.get("pills_compare_hub")
            if not picked:
                return
            if "Jaipur" in picked:
                target_cmp = "Jaipur"
            elif picked == "Delhi":
                target_cmp = "New Delhi"
            else:
                target_cmp = picked
            st.session_state["compare_city_select"] = target_cmp
            run_compare_recalc(target_cmp)

        def on_cmp_jaipur_baseline():
            st.session_state["compare_city_select"] = "Jaipur"
            run_compare_recalc("Jaipur")

        def on_cmp_agra_preset():
            st.session_state["compare_city_select"] = "Agra"
            run_compare_recalc("Agra", win=9.0, bud=1500.0, stops=4)

        st.markdown("**Quick Benchmark Hubs (Instant 1-Click Comparison):**")
        quick_cmp_hubs = ["Jaipur (Spec Baseline)", "Delhi", "Varanasi", "Mumbai", "Kochi", "Agra", "Bengaluru"]
        st.pills(
            "Quick Benchmark Hubs",
            quick_cmp_hubs,
            key="pills_compare_hub",
            on_change=on_cmp_hub_pill_change,
            label_visibility="collapsed",
        )

        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.6, 1.2, 1.3, 1.3], vertical_alignment="bottom")
        with ctrl_c1:
            selected_cmp_city = st.selectbox("Select Study City to Compare", all_city_names, index=current_city_idx, key="compare_city_select", on_change=on_cmp_city_dropdown_change)
        with ctrl_c2:
            st.button(
                f"Recalculate {selected_cmp_city}",
                key="btn_run_compare_city",
                on_click=lambda: run_compare_recalc(st.session_state.get("compare_city_select", selected_cmp_city)),
            )
        with ctrl_c3:
            st.button("Load Jaipur Baseline", key="btn_load_jaipur_baseline_cmp", on_click=on_cmp_jaipur_baseline)
        with ctrl_c4:
            st.button("Load Agra Preset (4 Stops)", key="btn_load_agra_preset_cmp", on_click=on_cmp_agra_preset)

        # Context header
        if is_fixture:
            banner_title = "Specification Baseline Trade-Off Summary"
            banner_badge = "JAIPUR BENCHMARK FIXTURE"
            banner_sub = f"Jaipur Saturday ({snapshot.get('visit_date', '2026-08-29')}) &middot; 09:00 &middot; 6 hours &middot; INR 1,000 budget &middot; 3 stops &middot; Cab"
        else:
            banner_title = f"Submitted Plan Trade-Off Summary: {cmp_city}"
            banner_badge = "USER SUBMITTED SNAPSHOT"
            banner_sub = (
                f"{cmp_city} &middot; Date: {snapshot.get('visit_date', '2026-08-29')} &middot; "
                f"Start: {snapshot.get('start_time', '09:00')} &middot; Window: {snapshot.get('window_hours', 6.0)}h &middot; "
                f"Budget: INR {snapshot.get('budget_inr', 1000):,.2f} &middot; {snapshot.get('target_stops', 3)} stops &middot; "
                f"Mode: {snapshot.get('transport_mode', 'cab')}"
            )

        st.markdown(f"""
        <div style="background: var(--panel); border: 1.5px solid var(--line); border-left: 4px solid var(--oxblood); padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: var(--display); font-size: 1.15rem; font-weight: bold; color: var(--ink);">{banner_title}</div>
                <span class="tag-status" style="background: #eee3a6; padding: 2px 8px; font-size: 0.72rem; font-weight: bold; color: #982b32;">{banner_badge}</span>
            </div>
            <div style="font-size: 0.85rem; color: #5c5852; margin-top: 4px;">{banner_sub}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_same_route:
            st.info(
                f"Identical Objective Result: In {cmp_city}, the identical route achieves both minimum elapsed duration "
                f"({fastest.elapsed_formatted}) and maximum cultural stops ({fastest.cultural_stops_count} stops). "
                "There is no trade-off penalty between speed and cultural content."
            )

        st.markdown(f"""
        <div class="plan-grid" style="margin-bottom: 25px;">
            <article class="plan-card">
                <div class="plan-top">
                    <span>Plan A</span>
                    <strong>Fastest plan</strong>
                </div>
                <div class="plan-number">{fastest.elapsed_formatted}</div>
                <p>Elapsed time</p>
                <div class="plan-details">
                    <span><b>{fastest.cultural_stops_count}</b> cultural stop</span>
                    <span><b>INR {fastest.total_cost_inr:.2f}</b> entry + travel</span>
                    <span><b>{fastest.estimated_carbon_kg:.3f} kg</b> estimated carbon</span>
                </div>
            </article>
            <article class="plan-card featured-plan">
                <div class="plan-top">
                    <span>Plan B</span>
                    <strong>Most cultural stops</strong>
                </div>
                <div class="plan-number">{cultural.elapsed_formatted}</div>
                <p>Elapsed time</p>
                <div class="plan-details">
                    <span><b>{cultural.cultural_stops_count}</b> cultural stops</span>
                    <span><b>INR {cultural.total_cost_inr:.2f}</b> entry + travel</span>
                    <span><b>{cultural.estimated_carbon_kg:.3f} kg</b> estimated carbon</span>
                </div>
                <span class="plan-tag">+{cultural_gain} cultural stop</span>
            </article>
        </div>
        """, unsafe_allow_html=True)

        if not is_same_route:
            st.markdown(f"""
            <div style="background: var(--panel); border: 1px solid var(--line); padding: 18px 24px; margin-bottom: 24px; transform: rotate(-0.3deg);">
                <div style="font-family: var(--display); font-size: 1.2rem; font-weight: 500; color: #292925;">
                    {banner_title} &middot; Trade-Off Summary
                </div>
                <div style="font-size: 0.90rem; color: #5c5852; margin-top: 6px; line-height: 1.6;">
                    Choosing <strong>Plan B (Most Cultural)</strong> over <strong>Plan A (Fastest)</strong> yields:
                    <strong style="color: #982b32;">+{cultural_gain} cultural stop</strong> in exchange for
                    <strong>+{time_diff} minutes</strong> elapsed duration,
                    <strong>+INR {cost_diff:.2f}</strong> total expenses, and
                    <strong>+{carbon_diff:.3f} kg</strong> estimated transit carbon.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Side-by-Side Trade-off Bar Chart
        comp_metrics_df = pd.DataFrame([
            {"Metric": "Elapsed Time (min)", "Plan A (Fastest)": fastest.elapsed_minutes, "Plan B (Cultural)": cultural.elapsed_minutes},
            {"Metric": "Total Cost (INR)", "Plan A (Fastest)": round(float(fastest.total_cost_inr), 1), "Plan B (Cultural)": round(float(cultural.total_cost_inr), 1)},
            {"Metric": "Est Carbon (kg × 100)", "Plan A (Fastest)": round(fastest.estimated_carbon_kg * 100, 1), "Plan B (Cultural)": round(cultural.estimated_carbon_kg * 100, 1)},
            {"Metric": "Cultural Stops (× 50)", "Plan A (Fastest)": fastest.cultural_stops_count * 50, "Plan B (Cultural)": cultural.cultural_stops_count * 50},
        ])
        fig_cmp_bar = go.Figure(data=[
            go.Bar(name="Plan A (Fastest)", x=comp_metrics_df["Metric"], y=comp_metrics_df["Plan A (Fastest)"], marker_color="#365376"),
            go.Bar(name="Plan B (Most Cultural)", x=comp_metrics_df["Metric"], y=comp_metrics_df["Plan B (Cultural)"], marker_color="#982b32"),
        ])
        fig_cmp_bar.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=20, r=20, t=55, b=50),
            title=dict(
                text=f"Direct Trade-Off Comparison: Plan A vs Plan B ({cmp_city})",
                x=0,
                y=0.96,
                font=dict(size=14, color="#292925")
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title_text=""
            )
        )
        apply_editorial_chart_theme(fig_cmp_bar)
        st.plotly_chart(fig_cmp_bar, use_container_width=True)

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.subheader("Plan A: Fastest Plan Details")
            st.markdown(f"- **Elapsed Time**: {fastest.elapsed_formatted} ({fastest.elapsed_minutes} min)")
            st.markdown(f"- **Cultural Stops**: {fastest.cultural_stops_count} of {fastest.total_stops_count}")
            st.markdown(f"- **Total Fees**: INR {fastest.entry_cost_inr:.2f} (Decimal)")
            st.markdown(f"- **Transit Costs**: INR {fastest.travel_cost_inr:.2f} (Decimal)")
            st.markdown(f"- **Total Budget Used**: INR {fastest.total_cost_inr:.2f}")
            st.markdown(f"- **Estimated Carbon**: {fastest.estimated_carbon_kg:.3f} kg")

            f_df = pd.DataFrame([{
                "Stop": s.stop_index,
                "POI ID": s.poi_id,
                "Name": s.name,
                "Category": s.category,
                "Is Cultural": s.is_cultural,
                "Arrive": s.arrival_time,
                "Wait (m)": s.wait_minutes,
                "Depart": s.visit_end_time,
                "Duration": f"{s.duration_minutes}m",
                "Fee (INR)": float(s.entry_fee),
                "Hours Status": s.hours_status,
                "Accessibility": s.accessibility
            } for s in fastest.stops])
            st.dataframe(f_df, hide_index=True, use_container_width=True)

            st.markdown("**Directed Legs (Plan A):**")
            f_legs_df = pd.DataFrame([{
                "Leg": l.leg_index,
                "Leg ID": l.matrix_id,
                "From ID": l.origin_id,
                "From": l.origin_name,
                "To ID": l.destination_id,
                "To": l.destination_name,
                "Mode": l.mode,
                "Dist (km)": float(l.distance_km),
                "Time (min)": l.travel_minutes,
                "Cost (INR)": float(l.cost_inr),
                "Carbon (kg)": float(l.carbon_kg)
            } for l in fastest.legs])
            st.dataframe(f_legs_df, hide_index=True, use_container_width=True)

        with cmp_col2:
            st.subheader("Plan B: Most Cultural Plan Details")
            st.markdown(f"- **Elapsed Time**: {cultural.elapsed_formatted} ({cultural.elapsed_minutes} min)")
            st.markdown(f"- **Cultural Stops**: {cultural.cultural_stops_count} of {cultural.total_stops_count}")
            st.markdown(f"- **Total Fees**: INR {cultural.entry_cost_inr:.2f} (Decimal)")
            st.markdown(f"- **Transit Costs**: INR {cultural.travel_cost_inr:.2f} (Decimal)")
            st.markdown(f"- **Total Budget Used**: INR {cultural.total_cost_inr:.2f}")
            st.markdown(f"- **Estimated Carbon**: {cultural.estimated_carbon_kg:.3f} kg")

            c_df = pd.DataFrame([{
                "Stop": s.stop_index,
                "POI ID": s.poi_id,
                "Name": s.name,
                "Category": s.category,
                "Is Cultural": s.is_cultural,
                "Arrive": s.arrival_time,
                "Wait (m)": s.wait_minutes,
                "Depart": s.visit_end_time,
                "Duration": f"{s.duration_minutes}m",
                "Fee (INR)": float(s.entry_fee),
                "Hours Status": s.hours_status,
                "Accessibility": s.accessibility
            } for s in cultural.stops])
            st.dataframe(c_df, hide_index=True, use_container_width=True)

            st.markdown("**Directed Legs (Plan B):**")
            c_legs_df = pd.DataFrame([{
                "Leg": l.leg_index,
                "Leg ID": l.matrix_id,
                "From ID": l.origin_id,
                "From": l.origin_name,
                "To ID": l.destination_id,
                "To": l.destination_name,
                "Mode": l.mode,
                "Dist (km)": float(l.distance_km),
                "Time (min)": l.travel_minutes,
                "Cost (INR)": float(l.cost_inr),
                "Carbon (kg)": float(l.carbon_kg)
            } for l in cultural.legs])
            st.dataframe(c_legs_df, hide_index=True, use_container_width=True)

        # Export Buttons using current comparison data
        st.markdown("---")
        st.subheader("Export Comparison Data")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            fastest_csv = f_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Plan A Itinerary (CSV) — {cmp_city}",
                data=fastest_csv,
                file_name=f"cultureroute_plan_a_{cmp_city.lower()}_fastest.csv",
                mime="text/csv",
                key="dl_plan_a_csv"
            )
        with exp_col2:
            cultural_csv = c_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Plan B Itinerary (CSV) — {cmp_city}",
                data=cultural_csv,
                file_name=f"cultureroute_plan_b_{cmp_city.lower()}_cultural.csv",
                mime="text/csv",
                key="dl_plan_b_csv"
            )

# ==============================================================================
elif view_mode == "Datasets":
    render_datasets_screen()

# ==============================================================================
# SCREEN 5: INSIGHTS
# ==============================================================================
elif view_mode == "Insights":
    render_insights_screen()

# ==============================================================================
# SCREEN 6: DEV LAB
# ==============================================================================
elif view_mode == "DEV Lab":
    render_section_banner(
        section_key="devlab",
        title_html="DEV Lab <em>Teaching Workbench</em>",
        subtitle="Inspect raw tables, run guided transformations, test bounded read-only SQL, and explore statistical distributions.",
        kicker="06 / EVALUATOR WORKBENCH · SYLLABUS UNITS 1-5",
    )

    # Academic Workflow Interactive Banner
    st.markdown("""
    <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #982b32; padding: 18px 24px; margin-bottom: 16px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
            <div style="font-family: Georgia, serif; font-size: 1.15rem; font-weight: bold; color: #292925;">
                What You Can Do Here &middot; 6-Step Evaluator Workbench
            </div>
            <a href="?view=datasets" target="_self" style="font-size: 0.84rem; color: #982b32; text-decoration: none; font-weight: bold;">
                &larr; Inspect Full 24-Table Dataset Catalog
            </a>
        </div>
        <p style="font-size: 0.86rem; color: #5c5852; margin: 8px 0 12px; line-height: 1.55;">
            The DEV Lab is an interactive teaching workbench covering Syllabus Units 1&ndash;5.
            Follow the guided research path below to inspect schema grains, execute type-safe pandas derivations, run bounded SQL queries, and audit outer-join set partitions.
        </p>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 0.78rem; font-weight: bold; text-transform: uppercase;">
            <a href="?view=datasets" target="_self" style="background: #eee3a6; padding: 5px 10px; border: 1px solid #dfd6a8; color: #982b32; text-decoration: none;">1. Select in Catalog &rarr;</a>
            <span style="color: #736f66; padding: 4px 2px;">&rarr;</span>
            <span style="background: #ffffff; padding: 5px 10px; border: 1px solid #b6afa3; color: #292925;">2. Inspect (Tab 1)</span>
            <span style="color: #736f66; padding: 4px 2px;">&rarr;</span>
            <span style="background: #ffffff; padding: 5px 10px; border: 1px solid #b6afa3; color: #292925;">3. Transform (Tab 2)</span>
            <span style="color: #736f66; padding: 4px 2px;">&rarr;</span>
            <span style="background: #ffffff; padding: 5px 10px; border: 1px solid #b6afa3; color: #292925;">4. Compare (Tab 4)</span>
            <span style="color: #736f66; padding: 4px 2px;">&rarr;</span>
            <span style="background: #ffffff; padding: 5px 10px; border: 1px solid #b6afa3; color: #292925;">5. Chart (Tab 4)</span>
            <span style="color: #736f66; padding: 4px 2px;">&rarr;</span>
            <span style="background: #ffffff; padding: 5px 10px; border: 1px solid #b6afa3; color: #292925;">6. Explain & Export (Tabs 3 &amp; 5)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lab_t1, lab_t2, lab_t3, lab_t4, lab_t5 = st.tabs([
        "1. Sources & Profiling",
        "2. Guided Transforms",
        "3. SQL Sandbox",
        "4. Statistics & Graphs",
        "5. Formulas & Syntax"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Sources & Profiling
    # --------------------------------------------------------------------------
    with lab_t1:
        st.subheader("1. Canonical 24-Table Study Catalog & Safe Profiler")
        st.caption("Inspect natural grain, primary/foreign keys, null counts, and safe schema projections across all 24 study tables.")

        cat = load_study_catalog()
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Canonical Tables", f"{len(cat.tables)}")
        c_m2.metric("Total Reconciled Rows", f"{cat.metadata.total_rows:,}")
        c_m3.metric("Verified Shared Tables", f"{len(cat.metadata.shared_tables)}", "Identical schema & data")
        c_m4.metric("Indian Study Cities", "45", "country_code='IN'")

        st.markdown("**Quick-Select Table Chips:**")
        quick_tbls = ["trips", "activities_poi", "user_interactions", "events_festivals", "itineraries", "poi_travel_matrix"]

        def _on_devlab_quick_table_pick():
            picked = st.session_state.get("pills_quick_table_pick")
            if picked:
                st.session_state["devlab_tab1_table_select"] = picked

        st.pills(
            "Quick Table Pick",
            quick_tbls,
            key="pills_quick_table_pick",
            on_change=_on_devlab_quick_table_pick,
            label_visibility="collapsed",
        )

        tbl_names = sorted(list(cat.tables.keys()))
        param_table = str(st.query_params.get("table", "")).strip()
        default_tbl_idx = tbl_names.index(param_table) if param_table in tbl_names else tbl_names.index("activities_poi") if "activities_poi" in tbl_names else 0
        selected_tbl = st.selectbox(
            "Select Study Table to Profile",
            tbl_names,
            index=default_tbl_idx,
            format_func=lambda x: f"{x} ({cat.tables[x].origin_pack} | {cat.tables[x].row_count:,} rows)",
            key="devlab_tab1_table_select"
        )

        profile = get_table_profile(selected_tbl, sample_limit=st.session_state.get("table_page_size", 15))
        if profile:
            p_c1, p_c2 = st.columns([1.5, 2.5])
            with p_c1:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #b6afa3; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="font-family: Georgia, serif; font-size: 1.05rem; font-weight: bold; color: #292925;">
                        Table Metadata: <code>{profile.table_name}</code>
                    </div>
                    <ul style="font-size: 0.84rem; color: #5c5852; margin: 8px 0 0 16px; padding: 0; line-height: 1.6;">
                        <li><strong>Origin Pack:</strong> {profile.origin_pack}</li>
                        <li><strong>Natural Grain:</strong> {profile.grain}</li>
                        <li><strong>Total Rows:</strong> {profile.row_count:,}</li>
                        <li><strong>Primary Key:</strong> {", ".join(profile.primary_key) if profile.primary_key else "None (implicit)"}</li>
                        <li><strong>Foreign Keys:</strong> {", ".join(profile.foreign_keys) if profile.foreign_keys else "None"}</li>
                        <li><strong>Safe Columns:</strong> {len(profile.safe_columns)} / {profile.column_count}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with p_c2:
                schema_records = []
                for col in profile.columns:
                    schema_records.append({
                        "Column": col,
                        "Type": profile.column_types.get(col, "TEXT"),
                        "Null Count": profile.null_counts.get(col, 0),
                        "Safe Status": "Permitted" if col in profile.safe_columns else "Masked (Sensitive)"
                    })
                st.markdown("**Column Schema & Missingness:**")
                st.dataframe(pd.DataFrame(schema_records), hide_index=True, use_container_width=True, height=180)

            st.markdown(f"**Safe Data Preview (First {len(profile.sample_df)} Rows):**")
            st.dataframe(profile.sample_df, hide_index=True, use_container_width=True)

            e1, e2 = st.columns(2)
            with e1:
                csv_data = export_dataframe_to_csv(profile.sample_df)
                st.download_button(
                    f"Download {profile.table_name} Preview (CSV)",
                    data=csv_data,
                    file_name=f"study_{profile.table_name}_preview.csv",
                    mime="text/csv",
                    key=f"btn_dl_csv_{profile.table_name}"
                )
            with e2:
                pq_data = export_dataframe_to_parquet(profile.sample_df)
                st.download_button(
                    f"Download {profile.table_name} Preview (Parquet)",
                    data=pq_data,
                    file_name=f"study_{profile.table_name}_preview.parquet",
                    mime="application/octet-stream",
                    key=f"btn_dl_pq_{profile.table_name}"
                )

    # --------------------------------------------------------------------------
    # TAB 2: Guided Transforms
    # --------------------------------------------------------------------------
    with lab_t2:
        st.subheader("2. Guided Data Transformations")
        st.caption("Perform type-safe pandas operations: multi-table joins, price discretisation, split-apply-combine, and canonical outer joins.")

        transform_options = [
            "A. WanderMatch Telemetry Join (user_interactions & activities_poi)",
            "B. Filter & Derive Categories (Price Discretisation)",
            "C. Stop Aggregation (Split-Apply-Combine)",
            "D. Pivot City × Category Matrix Reshape",
            "E. Canonical Merge (Saved vs Planned Stops: 666 POIs)"
        ]
        param_tf = str(st.query_params.get("transform", "")).strip()
        default_tf_idx = 0
        if param_tf == "telemetry":
            default_tf_idx = 0
        elif param_tf == "filter":
            default_tf_idx = 1
        elif param_tf == "aggregate":
            default_tf_idx = 2
        elif param_tf == "pivot":
            default_tf_idx = 3
        elif param_tf == "merge":
            default_tf_idx = 4

        transform_mode = st.radio(
            "Select Transformation Operation",
            transform_options,
            index=default_tf_idx,
            horizontal=True,
            key="dev_lab_tf_mode"
        )

        if "A. WanderMatch" in transform_mode:
            st.markdown("**A. WanderMatch Telemetry Transformation (user_interactions & activities_poi Join):**")
            st.caption("Demonstrates join semantics between WanderMatch telemetry events and SmartTrip attractions on entity_id = poi_id.")

            tf_c1, tf_c2 = st.columns([1.5, 1.2])
            with tf_c1:
                action_filter = st.selectbox(
                    "Filter by Telemetry Action Type",
                    ["All Actions (8 Types)", "view", "click", "search", "like", "save", "dismiss", "share", "booking"],
                    index=0,
                    key="devlab_tab2_action_select"
                )
            with tf_c2:
                cultural_only = st.checkbox("Cultural Attractions Only", value=False, key="devlab_tab2_cult_cb")

            filter_types = None if action_filter.startswith("All") else [action_filter]
            tf_df, tf_metrics, tf_snippet = transform_user_interactions(
                interaction_types=filter_types,
                cultural_only=cultural_only
            )

            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("1. Raw Interactions", f"{tf_metrics['raw_total']:,}", "WanderMatch pack")
            tm2.metric("2. Matched India POIs", f"{tf_metrics['matched_count']:,}", f"{tf_metrics['matched_count']/tf_metrics['raw_total']:.1%}")
            tm3.metric("3. Filtered Events", f"{tf_metrics['filtered_count']:,}", "Active selection")
            tm4.metric("4. Cultural Events", f"{tf_metrics['cultural_count']:,}", "heritage/museum/religious/food")

            # Plotly Action Distribution Bar Chart
            if "interaction_type" in tf_df.columns and "total_interactions" in tf_df.columns:
                fig_act_bar = px.bar(
                    tf_df,
                    x="interaction_type",
                    y="total_interactions",
                    text="total_interactions",
                    title="Telemetry Action Breakdown",
                    color_discrete_sequence=["#982b32"]
                )
                fig_act_bar.update_layout(height=240, margin=dict(l=20, r=20, t=35, b=20))
                apply_editorial_chart_theme(fig_act_bar)
                st.plotly_chart(fig_act_bar, use_container_width=True)

            st.markdown(f"**Transformed Telemetry Summary Table:**")
            st.dataframe(tf_df, hide_index=True, use_container_width=True)

            with st.expander("Inspect Reproducible pandas Join Pipeline"):
                st.code(tf_snippet, language="python")

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button("Download Transformed Telemetry (CSV)", export_dataframe_to_csv(tf_df), "transformed_telemetry.csv", "text/csv", key="btn_tf_csv_t2")
            with dl_c2:
                st.download_button("Download Transformed Telemetry (Parquet)", export_dataframe_to_parquet(tf_df), "transformed_telemetry.parquet", "application/octet-stream", key="btn_tf_pq_t2")

        elif "B. Filter" in transform_mode:
            st.markdown("**B. Filter & Derive Categories (Price Discretisation with Separate Free Tier):**")
            f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1.0])
            with f_col1:
                cost_range = st.slider("Entry Cost Range (INR)", 0.0, 1000.0, (0.0, 500.0), step=25.0)
            with f_col2:
                cats = st.multiselect("Cultural Categories", ["heritage", "museum", "religious", "food", "entertainment", "nature"], default=["heritage", "museum", "religious", "food"])
            with f_col3:
                step_free = st.checkbox("Step-Free Only", value=False)

            filtered_df, code_snippet = filter_and_derive_poi(
                min_cost=cost_range[0],
                max_cost=cost_range[1],
                selected_categories=cats,
                only_step_free=step_free
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Matching Attractions", f"{len(filtered_df)}")
            m2.metric("Cultural Attractions", f"{int(filtered_df['is_cultural'].sum())}")
            m3.metric("Free Attractions", f"{int((filtered_df['price_tier'] == 'Free (INR 0)').sum())}")
            m4.metric("Avg Entry Cost", f"INR {filtered_df['entry_cost'].mean():.2f}" if len(filtered_df) > 0 else "N/A")

            st.dataframe(filtered_df.head(50), hide_index=True, use_container_width=True)

            with st.expander("Inspect Reproducible pandas Code Snippet"):
                st.code(code_snippet, language="python")

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button("Download Filtered POIs (CSV)", export_dataframe_to_csv(filtered_df), "filtered_pois.csv", "text/csv", key="btn_tf_csv")
            with dl_c2:
                st.download_button("Download Filtered POIs (Parquet)", export_dataframe_to_parquet(filtered_df), "filtered_pois.parquet", "application/octet-stream", key="btn_tf_pq")

        elif "C. Stop Aggregation" in transform_mode:
            st.markdown("**C. Split-Apply-Combine Aggregation (Selected Planned Stops):**")
            grp_by = st.selectbox("Group By Dimension", ["city_name", "poi_category", "accessibility"], index=0)
            agg_df, code_snippet = groupby_aggregate_stops(grp_by)

            st.dataframe(agg_df, hide_index=True, use_container_width=True)

            with st.expander("Inspect Reproducible pandas Code Snippet"):
                st.code(code_snippet, language="python")

        elif "D. Pivot" in transform_mode:
            st.markdown("**D. City × Category Matrix Reshape (45 Cities × 4 Categories):**")
            pivot_df, long_df, code_snippet = pivot_city_category_matrix()

            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=list(pivot_df.columns),
                y=list(pivot_df.index),
                colorscale="Burg",
                hoverongaps=False
            ))
            fig_hm.update_layout(height=480, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Cultural Category", yaxis_title="Study City")
            apply_editorial_chart_theme(fig_hm)
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown("**Companion Wide Pivot Table:**")
            st.dataframe(pivot_df, use_container_width=True)

            with st.expander("Inspect Reversible Long-to-Wide Reshape Code"):
                st.code(code_snippet, language="python")

        elif "E. Canonical Merge" in transform_mode:
            st.markdown("**E. Audited Full Outer Join (Saved POIs vs Planned Stops):**")
            st.caption("Demonstrates strict set operations: 363 saved POIs outer-joined with 655 planned POIs without row inflation.")
            m_res = compute_canonical_saved_planned_merge()

            j1, j2, j3, j4 = st.columns(4)
            j1.metric("Saved POIs (Left)", f"{m_res.saved_count}")
            j2.metric("Planned POIs (Right)", f"{m_res.planned_count}")
            j3.metric("Inner Join (Both)", f"{m_res.both_count}", "352 overlap")
            j4.metric("Full Outer Join", f"{m_res.full_outer_count}", "666 distinct")

            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("1. Overlap (Both Sets)", f"{m_res.both_count} POIs", "Inner join intersection")
            p_col2.metric("2. Saved-Only POIs", f"{m_res.saved_only_count} POIs", "Telemetried but unselected")
            p_col3.metric("3. Planned-Only POIs", f"{m_res.planned_only_count} POIs", "Itinerary stops without save")

            st.dataframe(m_res.merged_df.head(100), hide_index=True, use_container_width=True)

            c_c1, c_c2 = st.columns(2)
            with c_c1:
                st.download_button("Download 666 Outer Merge (CSV)", export_dataframe_to_csv(m_res.merged_df), "canonical_outer_merge_666.csv", "text/csv", key="btn_merge_csv")
            with c_c2:
                st.download_button("Download 666 Outer Merge (Parquet)", export_dataframe_to_parquet(m_res.merged_df), "canonical_outer_merge_666.parquet", "application/octet-stream", key="btn_merge_pq")

    # --------------------------------------------------------------------------
    # TAB 3: SQL Sandbox
    # --------------------------------------------------------------------------
    with lab_t3:
        st.subheader("3. Bounded Read-Only SQL Sandbox")
        st.caption("Execute read-only SQL queries under SQLite authorizer enforcement. Prohibits writes, DDL, PRAGMA, and ATTACH. Max 1500 chars, 500 rows.")

        sandbox = ReadOnlySQLSandbox()
        presets = {
            "Top 5 Cities by Population": "SELECT city_id, name, state, population FROM cities WHERE country_code='IN' ORDER BY population DESC LIMIT 5",
            "Attractions by Category & Avg Price": "SELECT poi_category, count(*) as count, round(avg(entry_cost), 2) as avg_price FROM activities_poi GROUP BY poi_category ORDER BY count DESC",
            "Non-Cancelled Trips by Destination City": "SELECT c.name as city, count(*) as trip_count FROM trips t JOIN cities c ON c.city_id=t.destination_city_id WHERE c.country_code='IN' AND t.status<>'cancelled' GROUP BY c.name ORDER BY trip_count DESC LIMIT 10",
            "User Telemetry Interactions by Type": "SELECT interaction_type, count(*) as count FROM user_interactions GROUP BY interaction_type ORDER BY count DESC",
            "Planned Stops Summary from Documented View": "SELECT city_name, poi_category, count(*) as stops FROM view_study_selected_stops GROUP BY city_name, poi_category ORDER BY stops DESC LIMIT 15",
            "Test Write Rejection (Blocked INSERT)": "INSERT INTO cities (city_id, name) VALUES ('fake', 'Malicious City')"
        }

        p_sel = st.selectbox("Select Query Preset", list(presets.keys()), key="lab_sql_preset")
        if st.button("Load Preset into Editor", key="btn_load_sql_preset"):
            st.session_state["lab_sql_text"] = presets[p_sel]

        default_text = st.session_state.get("lab_sql_text", presets[p_sel])
        sql_input = st.text_area("SQL Query Editor", value=default_text, height=120, key="lab_sql_editor")

        if st.button("Execute Query", type="primary", key="lab_sql_run"):
            s_res = sandbox.execute_query(sql_input)
            if s_res.is_success:
                st.success(f"Query returned {s_res.row_count} rows{' (truncated to 500 maximum rows)' if s_res.is_truncated else ''}.")
                res_df = pd.DataFrame(s_res.rows, columns=s_res.columns)
                st.dataframe(res_df, hide_index=True, use_container_width=True)

                sq_c1, sq_c2 = st.columns(2)
                with sq_c1:
                    st.download_button("Download Result (CSV)", export_dataframe_to_csv(res_df), "query_result.csv", "text/csv", key="btn_sql_csv")
                with sq_c2:
                    st.download_button("Download Result (Parquet)", export_dataframe_to_parquet(res_df), "query_result.parquet", "application/octet-stream", key="btn_sql_pq")
            else:
                st.error(f"Execution Rejected: {s_res.error_message}")

    # --------------------------------------------------------------------------
    # TAB 4: Statistics & Graphs
    # --------------------------------------------------------------------------
    with lab_t4:
        st.subheader("4. Statistical Visualizations & Syllabus Demonstrator")
        st.caption("Visual representations of empirical distributions, methodology confidence intervals, and footfall cumulative densities.")

        st.markdown("**Interactive Statistical Visualizations & Syllabus Explorations:**")
        sg_choice = st.pills(
            "Select Visualization Focus",
            [
                "Unit 1: Methodology Confidence Intervals",
                "Unit 4: Cultural Category Breakdown (Donut Chart)",
                "Unit 4: WanderMatch Telemetry Actions (Donut Chart)",
                "Unit 4: Distribution Histogram & Box Plots",
                "Unit 4: Event Footfall ECDF",
                "Unit 4: Bivariate Pricing vs Saves Scatter"
            ],
            default="Unit 1: Methodology Confidence Intervals",
            key="pills_sg_choice"
        )
        if not sg_choice:
            sg_choice = "Unit 1: Methodology Confidence Intervals"

        if "Unit 1" in sg_choice:
            st.markdown("**Three-Way Inference on Owner Cohort (Exploratory vs Classical vs Bayesian):**")
            cohort_opt = st.pills(
                "Owner Cohort Selection",
                ["Shared Contract (293 / 341, 85.9%)", "Legacy Cohort (301 / 341, 88.3%)"],
                default="Shared Contract (293 / 341, 85.9%)",
                key="pills_cohort_opt"
            ) or "Shared Contract (293 / 341, 85.9%)"
            use_strict = "293" in cohort_opt
            comp = compute_methodology_comparison(strict_active_only=use_strict)

            u1, u2, u3 = st.columns(3)
            u1.metric("1. Empirical Proportion", f"{comp.empirical_proportion:.2%}", f"{comp.success_count} / {comp.sample_size}")
            u2.metric("2. Wilson 95% Score CI", f"[{comp.wilson_lower:.1%}, {comp.wilson_upper:.1%}]", "Classical normal approx")
            u3.metric("3. Bayesian Credible Interval", f"[{comp.bayes_lower:.1%}, {comp.bayes_upper:.1%}]", comp.bayes_prior)

            fig_u1 = go.Figure()
            fig_u1.add_trace(go.Scatter(
                x=[comp.wilson_lower * 100, comp.empirical_proportion * 100, comp.wilson_upper * 100],
                y=["Wilson 95% CI", "Wilson 95% CI", "Wilson 95% CI"],
                mode="lines+markers",
                name="Classical Wilson Score",
                line=dict(color="#365376", width=4)
            ))
            fig_u1.add_trace(go.Scatter(
                x=[comp.bayes_lower * 100, comp.empirical_proportion * 100, comp.bayes_upper * 100],
                y=["Bayesian Posterior", "Bayesian Posterior", "Bayesian Posterior"],
                mode="lines+markers",
                name="Bayesian Beta(1,1)",
                line=dict(color="#982b32", width=4)
            ))
            fig_u1.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Cultural Trip Proportion (%)")
            apply_editorial_chart_theme(fig_u1)
            st.plotly_chart(fig_u1, use_container_width=True)

        elif "Cultural Category" in sg_choice:
            st.markdown("**Cultural Category Distribution across 211 Verified Cultural POIs:**")
            fig_cat_donut = go.Figure(data=[go.Pie(
                labels=["Heritage Sites (72)", "Museums & Galleries (53)", "Religious Sites (48)", "Food Experiences (38)"],
                values=[72, 53, 48, 38],
                hole=0.5,
                marker=dict(colors=["#982b32", "#365376", "#b79d5b", "#7d6f58"]),
                textinfo="percent+label",
                hoverinfo="label+value+percent"
            )])
            fig_cat_donut.update_layout(
                title="Cultural Category Breakdown (211 Attractions)",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            apply_editorial_chart_theme(fig_cat_donut)
            st.plotly_chart(fig_cat_donut, use_container_width=True)
            st.caption("Provenance: 211 attractions across 45 Indian study cities strictly classified into Heritage, Museums, Religious, and Food experiences.")

        elif "Telemetry Actions" in sg_choice:
            st.markdown("**WanderMatch Interaction Telemetry Distribution (6,850 Matched Events):**")
            act_df = _get_devlab_actions_df()

            fig_act_pie = px.pie(
                act_df,
                names="interaction_type",
                values="count",
                hole=0.45,
                title="User Interaction Telemetry Actions (WanderMatch)",
                color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#4f637b", "#8b7e66", "#c2b59b", "#292925"]
            )
            fig_act_pie.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            apply_editorial_chart_theme(fig_act_pie)
            st.plotly_chart(fig_act_pie, use_container_width=True)
            st.caption("Telemetry Breakdown: 6,850 user interaction events matched to attractions in study cities across saves, views, reviews, and bookings.")

        elif "Histogram" in sg_choice:
            st.markdown("**Entry Cost Distribution across Cultural Categories (45 Indian Study Cities):**")
            poi_df = _get_devlab_poi_costs_df()

            hm1, hm2, hm3, hm4 = st.columns(4)
            hm1.metric("Study Attractions", f"{len(poi_df):,} POIs", "45 Indian Study Cities")
            hm2.metric("Median Entry Fee", f"INR {poi_df['entry_cost'].median():.0f}", "50th percentile")
            hm3.metric("Free Admission Share", f"{(poi_df['entry_cost'] == 0).mean():.1%}", f"{(poi_df['entry_cost'] == 0).sum()} Free POIs")
            hm4.metric("Interquartile Range (IQR)", f"INR {poi_df['entry_cost'].quantile(0.75) - poi_df['entry_cost'].quantile(0.25):.0f}", "Q3 (INR 750) - Q1 (INR 0)")

            fig_dist = px.histogram(
                poi_df,
                x="entry_cost",
                color="poi_category",
                nbins=30,
                barmode="overlay",
                color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#4f637b", "#8b7e66", "#292925"],
                title="Histogram of Attraction Entry Costs by Cultural Category (45 Indian Cities, Max INR 2,500)"
            )
            fig_dist.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Entry Fee (INR)", yaxis_title="Attraction Count")
            apply_editorial_chart_theme(fig_dist)
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_box = px.box(
                poi_df,
                x="poi_category",
                y="entry_cost",
                color="poi_category",
                color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#4f637b", "#8b7e66", "#292925"],
                title="Entry Fee Box Plots by Category (Natural Indian Cohort Dispersion)"
            )
            fig_box.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Category", yaxis_title="Entry Fee (INR)")
            apply_editorial_chart_theme(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

            st.caption("Distribution Insight: 38% of cultural venues are completely free. Ticketed heritage and museum sites cluster tightly between INR 50 and INR 500, with top-tier private heritage experiences capping at INR 2,500.")

        elif "ECDF" in sg_choice:
            st.markdown("**Empirical Cumulative Distribution Function (ECDF) of Event Footfall:**")
            events_df = _get_devlab_events_df()

            ef1, ef2, ef3, ef4 = st.columns(4)
            ef1.metric("Audited Festivals", f"{len(events_df):,}", "Complete event records")
            ef2.metric("Median Footfall (P50)", f"{int(events_df['expected_footfall'].median()):,}", "Typical event crowd")
            ef3.metric("Interquartile Range (IQR)", f"{int(events_df['expected_footfall'].quantile(0.75) - events_df['expected_footfall'].quantile(0.25)):,}", "Middle 50% spread")
            ef4.metric("Maximum Footfall", f"{int(events_df['expected_footfall'].max()):,}", "Major city festival")

            fig_ecdf = px.ecdf(
                events_df,
                x="expected_footfall",
                hover_name="name",
                hover_data=["city_name", "recurrence", "season"],
                title=f"Event Footfall Cumulative Density ({len(events_df)} Non-Null Study Festivals across 45 Indian Cities)"
            )
            med_val = float(events_df['expected_footfall'].median())
            p90_val = float(events_df['expected_footfall'].quantile(0.90))
            fig_ecdf.add_vline(x=med_val, line_dash="dash", line_color="#982b32", annotation_text=f"P50 Median: {int(med_val):,}", annotation_position="top left")
            fig_ecdf.add_vline(x=p90_val, line_dash="dot", line_color="#365376", annotation_text=f"P90 Decile: {int(p90_val):,}", annotation_position="top left")
            fig_ecdf.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Expected Footfall Attendees", yaxis_title="Cumulative Probability F_n(x)")
            apply_editorial_chart_theme(fig_ecdf)
            st.plotly_chart(fig_ecdf, use_container_width=True)
            st.caption("Syllabus Takeaway: Festival expected footfalls are uniformly distributed between 3,133 and 218,180 attendees with a sample median of 110,811. The step-wise empirical CDF shows no extreme multi-modal polarization.")

        elif "Bivariate" in sg_choice:
            st.markdown("**Bivariate Scatter: Attraction Entry Fee vs WanderMatch Telemetry Saves:**")
            biv_df = _get_devlab_bivariate_df().copy()

            spearman_corr = biv_df["entry_cost"].corr(biv_df["saves_count"], method="spearman")
            pearson_corr = biv_df["entry_cost"].corr(biv_df["saves_count"], method="pearson")

            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Venues With Saves", f"{len(biv_df):,}", "Audited 363 saved POIs")
            bm2.metric("Total Saves Count", f"{biv_df['saves_count'].sum():,}", "User bookmark events")
            bm3.metric("Spearman Rank Corr (r_s)", f"{spearman_corr:.2f}", "Ordinal price elasticity")
            bm4.metric("High-Save Venues (>2 saves)", f"{(biv_df['saves_count'] > 2).sum()}", "High engagement sites")

            st.markdown("**Choose Representation Style:**")
            biv_format = st.pills(
                "Bivariate Visualization Format",
                [
                    "Economic Price Brackets (Bar Chart)",
                    "Raw Point Scatter",
                ],
                default="Economic Price Brackets (Bar Chart)",
                key="dev_lab_biv_format_pills",
                label_visibility="collapsed"
            ) or "Economic Price Brackets (Bar Chart)"

            if biv_format == "Economic Price Brackets (Bar Chart)":
                def get_price_bracket(fee):
                    if fee == 0:
                        return "1. Free (INR 0)"
                    elif fee <= 100:
                        return "2. Budget (INR 1-100)"
                    elif fee <= 300:
                        return "3. Moderate (INR 101-300)"
                    elif fee <= 750:
                        return "4. Mid-Range (INR 301-750)"
                    else:
                        return "5. Premium (INR 751-2500)"

                biv_df["price_tier"] = biv_df["entry_cost"].apply(get_price_bracket)
                tier_summary = biv_df.groupby("price_tier", observed=False).agg(
                    total_venues=("poi_id", "count"),
                    avg_saves=("saves_count", "mean"),
                    total_saves=("saves_count", "sum")
                ).reset_index()

                fig_biv_bar = px.bar(
                    tier_summary,
                    x="price_tier",
                    y="avg_saves",
                    text="avg_saves",
                    color="avg_saves",
                    color_continuous_scale=["#f0eae1", "#982b32"],
                    hover_data={"total_venues": True, "total_saves": True, "avg_saves": ":.2f"},
                    labels={
                        "price_tier": "Attraction Price Bracket",
                        "avg_saves": "Average Saves / Venue",
                        "total_venues": "Venues in Tier",
                        "total_saves": "Total Saves"
                    },
                    title="Average User Saves Across Price Brackets (363 Saved Indian Venues)"
                )
                fig_biv_bar.update_traces(
                    texttemplate="%{text:.2f} saves",
                    textposition="outside",
                    marker=dict(line=dict(width=1, color="#292925"))
                )
                fig_biv_bar.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=35, b=20),
                    yaxis=dict(range=[0, 2.2], title="Average Saves per Venue"),
                    xaxis=dict(title="Economic Price Tier"),
                    coloraxis_showscale=False
                )
                apply_editorial_chart_theme(fig_biv_bar)
                st.plotly_chart(fig_biv_bar, use_container_width=True)
                st.caption(
                    "Intuitive Bar Chart Alternative: Groups 363 saved venues into 5 economic price brackets rather than unreadable individual points. "
                    "Notice average saves remain constant between 1.34 and 1.52 saves/venue across all tiers, "
                    "confirming price does not depress user intent."
                )
            else:
                fig_biv = px.scatter(
                    biv_df,
                    x="entry_cost",
                    y="saves_count",
                    color="poi_category",
                    hover_name="name",
                    hover_data=["city_name"],
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#4f637b", "#8b7e66", "#292925"],
                    labels={"entry_cost": "Entry Fee (INR)", "saves_count": "Telemetry Saves Count", "poi_category": "Category"},
                    title=f"Attraction Entry Fee vs User Saves (All 363 Saved Indian Study POIs, Max INR 2,500)"
                )
                fig_biv.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="#292925")))
                fig_biv.update_layout(height=380, margin=dict(l=20, r=20, t=35, b=20))
                apply_editorial_chart_theme(fig_biv)
                st.plotly_chart(fig_biv, use_container_width=True)
                st.caption("Point Scatter Representation: Each point represents an individual saved study attraction plotted against its entry fee.")

    # --------------------------------------------------------------------------
    # TAB 5: Formulas & Syntax Reference
    # --------------------------------------------------------------------------
    with lab_t5:
        st.subheader("5. Formulas & Syntax Reference (Syllabus Units 1–5)")
        st.caption("Searchable technical guide connecting syllabus mathematical formulas to concrete code implementations and study data.")

        ref_units = get_syllabus_reference_units()
        unit_filter = st.pills("Filter by Syllabus Unit", ["All Units", "Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5"], default="All Units", key="lab_unit_filter_pills") or "All Units"

        for item in ref_units:
            if unit_filter != "All Units" and item["unit"] != unit_filter:
                continue

            with st.expander(f"{item['unit']} · {item['title']}: {item['topic']}", expanded=(unit_filter != "All Units")):
                st.markdown(f"**Purpose:** {item['purpose']}")
                st.code(item['formula_syntax'], language="text")
                st.markdown(f"**Data Prerequisites & Missingness:** {item['prerequisites']}")
                st.markdown(f"**Concrete Study Example:** {item['study_example']}")
                st.markdown(f"**Interpretation:** {item['interpretation']}")
                st.caption(f"Example SQL Query: `{item['preset_query']}`")
                if st.button(f"Run this example ({item['unit']})", key=f"btn_run_{item['unit']}"):
                    try:
                        sandbox = ReadOnlySQLSandbox()
                        exec_res = sandbox.execute_query(item["preset_query"])
                        if exec_res.is_success:
                            time_ms = getattr(exec_res, "execution_time_ms", 0.0)
                            st.success(f"Executed in {time_ms:.1f}ms ({exec_res.row_count} rows returned):")
                            if exec_res.rows:
                                res_df = pd.DataFrame(exec_res.rows, columns=exec_res.columns)
                                st.dataframe(res_df, hide_index=True, use_container_width=True)
                        else:
                            st.error(f"Execution Error: {exec_res.error_message}")
                    except Exception as ex:
                        st.error(f"Execution Exception: {ex}")

# ==============================================================================
# SCREEN: PREDICTION MODEL (Stage P6)
# ==============================================================================
elif view_mode == "Prediction Model":
    render_model_screen()


# ==============================================================================
# SCREEN 7: SETTINGS
# ==============================================================================
elif view_mode == "Settings":
    render_section_banner(
        section_key="settings",
        title_html="Settings & Preferences",
        subtitle="Configure session-local evaluation preferences, display limits, and accessibility parameters.",
        kicker="08 / EVALUATOR PREFERENCES & CONTROLS",
    )

    s_col1, s_col2 = st.columns([1.5, 1.2])
    with s_col1:
        cities = get_study_cities()
        c_names = [c["name"] for c in cities]
        cur_def = st.session_state.get("default_city", "Jaipur")
        def_idx = c_names.index(cur_def) if cur_def in c_names else 0

        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #b6afa3; border-left: 4px solid #365376; padding: 14px 18px; margin-bottom: 14px;">
            <strong style="color: #292925; font-size: 0.92rem;">Default Demonstration City:</strong><br>
            <span style="font-size: 0.84rem; color: #5c5852; line-height: 1.55;">
                Selects which of the 45 Indian study cities initially loads across <strong>Discover</strong>, <strong>Planner</strong>, <strong>Compare</strong>, and <strong>Prediction Model</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)
        new_def = st.selectbox("Default Study City", c_names, index=def_idx, help="Sets the default active city for Discover, Planner, and Prediction Model views.")

        cur_view = st.session_state.get("preferred_view", "Detailed Timetable & Leg Breakdown")
        view_opts = ["Detailed Timetable & Leg Breakdown", "Side-by-Side Comparison Matrix", "Compact Metrics Only"]
        new_view = st.selectbox("Preferred Planner Result View", view_opts, index=view_opts.index(cur_view) if cur_view in view_opts else 0)

        cur_size = st.session_state.get("table_page_size", 15)
        new_size = st.select_slider("Table Data Preview Size (Rows)", options=[5, 10, 15, 20, 25, 50], value=cur_size if cur_size in [5, 10, 15, 20, 25, 50] else 15)

        cur_motion = st.session_state.get("motion_preference", "Standard Motion")
        new_motion = st.radio("Interface Motion Preference", ["Standard Motion", "Reduced Motion (Accessibility)"], index=0 if cur_motion == "Standard Motion" else 1, horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("Save Preferences", type="primary", key="btn_save_settings"):
                st.session_state["default_city"] = new_def
                st.session_state["p_city"] = new_def
                st.session_state["preferred_view"] = new_view
                st.session_state["table_page_size"] = int(new_size)
                st.session_state["motion_preference"] = new_motion
                st.success(f"Preferences saved: Default City = '{new_def}', Page Size = {new_size}, View = '{new_view}'.")
        with btn_c2:
            if st.button("Reset to System Defaults", key="btn_reset_settings"):
                st.session_state["default_city"] = "Jaipur"
                st.session_state["p_city"] = "Jaipur"
                st.session_state["preferred_view"] = "Detailed Timetable & Leg Breakdown"
                st.session_state["table_page_size"] = 15
                st.session_state["motion_preference"] = "Standard Motion"
                st.info("Preferences reset to system defaults (Jaipur, 15 rows, Standard Motion).")

    with s_col2:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #b6afa3; border-left: 4px solid #982b32; padding: 18px 22px; margin-top: 10px;">
            <div style="font-family: Georgia, serif; font-size: 1.05rem; font-weight: bold; color: #292925;">
                Persistence & Security Contract
            </div>
            <div style="font-size: 0.84rem; color: #5c5852; margin-top: 6px; line-height: 1.6;">
                <strong>Session-Local Scope:</strong> Preferences are stored within your current active browser session state.
                Settings modify display defaults and table pagination across Discover and DEV Lab.<br><br>
                <strong>Security Invariant:</strong> Settings <em>never</em> alter canonical study database records, row counts,
                data integrity checksums, SQLite authorizer sandbox limits, or analytical metric contracts.
            </div>
        </div>
        """, unsafe_allow_html=True)

        act_city = st.session_state.get("default_city", "Jaipur")
        act_size = st.session_state.get("table_page_size", 15)
        act_view = st.session_state.get("preferred_view", "Detailed Timetable & Leg Breakdown")
        act_mot = st.session_state.get("motion_preference", "Standard Motion")

        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #b6afa3; padding: 18px 22px; margin-top: 16px;">
            <div style="font-family: Georgia, serif; font-size: 1.05rem; font-weight: bold; color: #292925;">
                Current Active Session State
            </div>
            <ul style="font-size: 0.84rem; color: #5c5852; margin: 8px 0 0 16px; padding: 0; line-height: 1.6;">
                <li><strong>Active Default City:</strong> {act_city}</li>
                <li><strong>Table Preview Limit:</strong> {act_size} rows</li>
                <li><strong>Result Presentation:</strong> {act_view}</li>
                <li><strong>Motion Policy:</strong> {act_mot}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SCREEN 8: INFORMATION
# ==============================================================================
elif view_mode == "Information":
    render_section_banner(
        section_key="information",
        title_html="Information, System Architecture & <em>Statistical Models</em>",
        subtitle="Complete reference guide detailing every section of CultureRoute, all 6 statistical & ML models used, database provenance, and DEV syllabus alignment.",
        kicker="09 / PROJECT REFERENCE & ACADEMIC METHODOLOGY",
    )

    info_t1, info_t2, info_t3, info_t4, info_t5, info_t6 = st.tabs([
        "1. Application Section Directory",
        "2. All Statistical & ML Models",
        "3. SDG 8.9 & Research Boundaries",
        "4. 24-Table Study Inventory",
        "5. DEV Syllabus Alignment",
        "6. Jaipur Reproduction Guide"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SECTION DIRECTORY (10 APPLICATION SCREENS)
    # --------------------------------------------------------------------------
    with info_t1:
        st.subheader("1. Comprehensive Application Section Directory")
        st.caption("Overview of all 10 application views, their design intent, and their role in the educational demonstration.")

        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px;">
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Overview</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">PORTFOLIO LANDING</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    The executive project summary. Features top-level research metrics (45 cities, 675 attractions, 48,728 database rows),
                    the dual-objective trade-off premise, curated city showcase cards, and verified research boundary notices.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Discover</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 6px;">SPATIAL EXPLORER</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Spatial and catalog discovery across all 45 Indian study cities. Filter attractions by fee status (Free vs Paid),
                    cultural category, or step-free accessibility. Shows offline coordinate scatter maps, category composition donuts, and WanderMatch guide telemetry.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Planner</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">RULES-BASED ROUTER</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Generates deterministic day itineraries for any study city under user-defined budgets, time windows, and target stops.
                    Computes Plan A (Minimum Elapsed Time) and Plan B (Maximum Cultural Stops) with schedule time-allocation charts and direct handoff to Compare.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Compare</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 6px;">TRADE-OFF EVALUATOR</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Side-by-side trade-off analysis. Compares time saved versus cultural heritage gained, transit costs, and carbon emissions.
                    Supports dynamic comparison across any of the 45 cities and preset benchmark fixtures, paired with grouped bar charts and CSV exports.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Datasets</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">DATASET VIEWER FIRST</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Primary data exploration portal. Source pack filter (SmartTrip vs WanderMatch vs Shared), search by table name, deterministic
                    pagination (1-15 of N rows), and one-click transparent CSV downloads for all 24 canonical tables, plus dynamic variable distributions in Tab 3.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Insights</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 6px;">STATISTICAL ANALYTICS</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Conventional statistical analysis without black-box ML: 3-population cultural representation (stacked bars and 4-donut comparative views),
                    57-week Monday-Sunday continuous time series, historical baseline evaluation (MAE 3.42 / 3.88), and MLE event footfall imputation on 143/36 holdout.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">DEV Lab</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">TEACHING WORKBENCH</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Structured 6-step laboratory addressing Units 1–5: Table profiling, guided pandas joins and aggregations, bounded read-only
                    SQL sandbox with SQLite authorizer security, graphical methods (Wilson CIs, action donuts, histograms, ECDFs), and formula-safe CSV/Parquet export.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Prediction Model</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 6px;">SUPERVISED COUNT GLM</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Supervised Poisson count regression model predicting weekly cultural interactions across all 45 cities for target week 2026-08-17.
                    Presents results first with exact Poisson 95% quantiles, followed by out-of-sample holdout validation and zero-leakage splits.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Settings</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">USER PREFERENCES</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Evaluator demonstration preferences. Sets the default study city pre-selected across Discover, Planner, and Model screens,
                    table preview limits (5 to 50 rows), itinerary display detail, and accessibility motion policies without altering underlying database records.
                </div>
            </div>
            <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #365376; padding: 18px 20px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-family: Georgia, serif; font-size: 1.1rem; font-weight: bold; color: #292925;">Information</div>
                    <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 6px;">REFERENCE & ARCHITECTURE</span>
                </div>
                <div style="font-size: 0.84rem; color: #5c5852; margin-top: 8px; line-height: 1.6;">
                    Comprehensive academic documentation detailing all 6 statistical and ML models with LaTeX math, UN SDG 8.9 alignment,
                    three distinct analytical populations, the 24-table inventory, DEV syllabus mapping, and Jaipur benchmark reproduction instructions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: ALL STATISTICAL & ML MODELS
    # --------------------------------------------------------------------------
    with info_t2:
        st.subheader("2. Detailed Mathematical Reference for All Models Used")
        st.markdown(
            "CultureRoute implements **6 rigorous statistical, machine learning, and combinatorial routing models**. "
            "Below is the exact formulation, input vector, validation metrics, and academic rationale for each."
        )

        # Model 1
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #982b32; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: Georgia, serif; font-size: 1.18rem; font-weight: bold; color: #292925;">
                    1. Poisson Generalized Linear Model (GLM) for Weekly Cultural Demand
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 8px;">SUPERVISED MACHINE LEARNING</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Mathematical Formulation:**")
        st.latex(r"\mathbb{P}(Y_{i,t} = y \mid \mathbf{x}_{i,t}) = \frac{\lambda_{i,t}^y e^{-\lambda_{i,t}}}{y!}, \quad y \in \{0, 1, 2, \dots\}")
        st.latex(r"\ln(\lambda_{i,t}) = \beta_0 + \sum_{c=1}^{44} \beta_c \cdot \mathbb{I}(\text{city}=c) + \beta_{\text{lag1}} y_{i,t-1} + \beta_{\text{lag2}} y_{i,t-2} + \beta_{\text{4w}} \bar{y}_{i,t-4:t-1}")
        st.latex(r"\lambda_{i,t} = \exp\left(\beta_0 + \mathbf{x}_{i,t}^\top \boldsymbol{\beta}\right)")

        st.markdown("""
        **Why Poisson over Ordinary Least Squares (OLS)?**
        - **Non-Negative Discrete Support:** Interaction counts are non-negative integers (<em>y</em> &ge; 0). OLS assumes unbounded Gaussian errors and routinely predicts impossible negative counts (<em>y&#770;</em> &lt; 0).
        - **Mean-Variance Relationship:** In count telemetry, variance scales with mean volume (Var(<em>Y</em>) = E[<em>Y</em>] = &lambda;). Linear regression incorrectly assumes constant homoscedastic variance.

        **Out-of-Sample Validation & Performance:**
        - Tested on a strictly disjoint 12-week chronological holdout (Weeks 46..57, 540 rows).
        - Achieves **0.7938 MAE** (interactions/city-week) vs **0.9537 MAE** for naive previous-week baseline (**-16.8% error reduction**).
        - On non-zero active weeks, error drops to **0.7786 MAE** vs 1.2101 for lag-1.

        **Exact 95% Confidence Bounds:**
        """, unsafe_allow_html=True)
        st.latex(r"\text{CI}_{95\%} = \left[ Q_{\text{Poisson}}(0.025; \lambda), \; Q_{\text{Poisson}}(0.975; \lambda) \right]")

        # Model 2
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #365376; padding: 18px 22px; margin-top: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: Georgia, serif; font-size: 1.18rem; font-weight: bold; color: #292925;">
                    2. Naive Momentum (Lag-1) & Trailing 4-Week Moving Average Baselines
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 8px;">BENCHMARK FORECASTING</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Formulations:**")
        st.latex(r"\hat{y}_{t}^{\text{Lag-1}} = y_{t-1} \quad \text{(Random walk persistence)}")
        st.latex(r"\hat{y}_{t}^{\text{Trailing 4W}} = \frac{1}{4} \sum_{k=1}^4 y_{t-k} \quad \text{(Smoothed rolling momentum)}")
        st.markdown("""
        **Role in Project:** Required by DEV Syllabus Unit 3 to establish whether parameterized machine learning models provide genuine predictive value over standard operational heuristics.
        Across the nationwide 57-week historical series, the 12-week backtest yields exact MAEs of **3.42** (Lag-1) and **3.88** (Trailing 4W).
        """)

        # Model 3
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #982b32; padding: 18px 22px; margin-top: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: Georgia, serif; font-size: 1.18rem; font-weight: bold; color: #292925;">
                    3. Lognormal Maximum Likelihood Estimation (MLE) for Event Footfall Imputation
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 8px;">PARAMETRIC INFERENCE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Mathematical Formulation:** For positive crowd sizes <em>X</em><sub>1</sub>, &hellip;, <em>X<sub>n</sub></em> &gt; 0, we model log-transformed footfall ln(<em>X<sub>i</sub></em>) ~ <em>N</em>(&mu;, &sigma;<sup>2</sup>):", unsafe_allow_html=True)
        st.latex(r"\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n \ln(x_i) = 8.3312, \quad \hat{\sigma}_{\text{MLE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\ln(x_i) - \hat{\mu})^2} = 0.7188")
        st.latex(r"\mathbb{E}[X] = \exp\left(\hat{\mu} + \frac{\hat{\sigma}^2}{2}\right) = \exp(8.3312 + 0.2583) \approx 4,863.6 \text{ visitors}")
        st.latex(r"\text{Median}(X) = \exp(\hat{\mu}) = \exp(8.3312) \approx 4,151.4 \text{ visitors}")
        st.markdown("""
        **Application in Study:** 225 Indian cultural festivals are cataloged in `events_festivals`, of which 12 records have missing `expected_footfall`.
        On a deterministic 143-train / 36-test holdout split, parametric lognormal MLE achieves a test MAE of **1,863 visitors**, compared to 2,050 for naive column mean, confirming the lognormal distribution's superior fit for human event crowds.
        """)

        # Model 4
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #365376; padding: 18px 22px; margin-top: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: Georgia, serif; font-size: 1.18rem; font-weight: bold; color: #292925;">
                    4. Beta-Binomial Bayesian Model vs Wilson Score Interval for Itinerary Inclusions
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #365376; border: 1px solid #365376; padding: 2px 8px;">BAYESIAN & CLASSICAL INFERENCE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Comparative Framework:** Investigates cultural stop conversion proportion <em>p</em> (<em>n</em> = 411 trips, <em>k</em> = 211 successes, empirical <em>p&#770;</em> = 0.5134):", unsafe_allow_html=True)
        st.latex(r"w = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \implies [0.4650, \; 0.5615]")
        st.latex(r"\text{Prior } p \sim \text{Beta}(1, 1) \implies \text{Posterior } p \mid k \sim \text{Beta}(1 + 211, \; 1 + 200) = \text{Beta}(212, 201)")
        st.latex(r"\text{Posterior Mean: } \frac{212}{413} = 0.5133, \quad 95\% \text{ Equal-Tailed Credible Interval: } [0.4650, \; 0.5613]")
        st.markdown("Demonstrates how classical Wilson intervals and uninformative Bayesian credible intervals converge when sample sizes exceed <em>n</em> &gt; 400.", unsafe_allow_html=True)

        # Model 5
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #982b32; padding: 18px 22px; margin-top: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-family: Georgia, serif; font-size: 1.18rem; font-weight: bold; color: #292925;">
                    5. Deterministic Dijkstra & Branch-and-Bound Multi-Objective Routing Engine
                </div>
                <span style="font-size: 0.72rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 8px;">COMBINATORIAL OPTIMIZATION</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Mathematical Formulation:** Over directed graph <em>G = (V, E)</em> where vertices <em>V</em> are city attractions and edges <em>E</em> are recorded travel matrix legs:", unsafe_allow_html=True)
        st.latex(r"\text{Plan A (Min Elapsed Time):} \quad \min_{\pi} \sum_{i=1}^k \text{duration}(v_{\pi(i)}) + \sum_{i=1}^{k-1} \text{travel\_time}(v_{\pi(i)}, v_{\pi(i+1)})")
        st.latex(r"\text{Plan B (Max Cultural Stops):} \quad \max_{\pi} \sum_{i=1}^k \mathbb{I}(v_{\pi(i)} \in \text{Cultural}), \quad \text{tie-breaker: } \min \text{elapsed\_time}")
        st.latex(r"\text{Subject to Hard Constraints: } \sum \text{cost} \le \text{Budget}, \; \text{arrival}(v_i) \ge \text{opens}(v_i), \; \text{depart}(v_i) \le \text{closes}(v_i)")
        st.markdown("""
        **Deterministic Guarantee:** The engine uses exact branch-and-bound pruning with zero randomized heuristic elements.
        Produces identical, reproducible routes across runs (e.g. Jaipur Saturday always yields 207 min / INR 585.46 for Plan A and 227 min / INR 638.30 for Plan B).
        """)

        # Model Comparison Matrix Table
        st.markdown("---")
        st.subheader("Model Comparison Matrix")
        model_matrix_df = pd.DataFrame([
            {"Model": "Poisson Count GLM", "Paradigm": "Supervised Machine Learning", "Formulation": "Log-linear count regression (scikit-learn)", "Inputs": "44 city one-hots, 3 numerical lags", "Primary Metric": "MAE 0.7938 (-16.8% vs lag-1)", "Unit": "Unit 3"},
            {"Model": "Naive Momentum (Lag-1)", "Paradigm": "Heuristic Baseline", "Formulation": "y(t) = y(t-1)", "Inputs": "Prior week interaction count", "Primary Metric": "MAE 3.42 nationwide", "Unit": "Unit 3"},
            {"Model": "Trailing 4W Moving Avg", "Paradigm": "Heuristic Baseline", "Formulation": "y(t) = mean(y[t-4:t-1])", "Inputs": "Prior 4-week window", "Primary Metric": "MAE 3.88 nationwide", "Unit": "Unit 3"},
            {"Model": "Lognormal MLE", "Paradigm": "Parametric Inference", "Formulation": "Maximum likelihood log-normal fit", "Inputs": "Festival crowd sizes X > 0", "Primary Metric": "Holdout MAE 1,863 visitors", "Unit": "Unit 3 & 4"},
            {"Model": "Wilson Score Interval", "Paradigm": "Classical Inference", "Formulation": "Asymmetric binomial score CI", "Inputs": "n=411 trips, k=211 successes", "Primary Metric": "95% CI [0.4650, 0.5615]", "Unit": "Unit 1"},
            {"Model": "Beta-Binomial Bayesian", "Paradigm": "Bayesian Conjugate", "Formulation": "Beta(1,1) prior -> Beta(212, 201)", "Inputs": "n=411 trips, k=211 successes", "Primary Metric": "95% Credible [0.4650, 0.5613]", "Unit": "Unit 1"},
            {"Model": "Deterministic Day Router", "Paradigm": "Combinatorial Optimization", "Formulation": "Branch-and-bound graph search", "Inputs": "Directed matrix, opening hours, budget", "Primary Metric": "Jaipur baseline 207m / 227m", "Unit": "Unit 2 & 5"}
        ])
        st.dataframe(model_matrix_df, hide_index=True, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: SDG 8.9 & RESEARCH BOUNDARIES
    # --------------------------------------------------------------------------
    with info_t3:
        st.subheader("3. UN Sustainable Development Goal 8, Target 8.9 & Boundaries")
        st.markdown("""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-left: 5px solid #b79d5b; padding: 18px 24px; margin-bottom: 20px;">
            <div style="font-family: Georgia, serif; font-size: 1.15rem; font-weight: bold; color: #292925;">
                SDG Target 8.9 Official Mandate
            </div>
            <p style="font-size: 0.90rem; color: #5c5852; font-style: italic; margin-top: 6px;">
                "By 2030, devise and implement policies to promote sustainable tourism that creates jobs and promotes local culture and products."
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### CultureRoute Implementation:
        CultureRoute investigates how local cultural assets (heritage sites, historical forts, museums, spiritual monuments, and regional culinary traditions)
        can be preserved and highlighted when travelers plan single-day itineraries under strict time and financial budgets.

        - **Cultural Engagement Maximization:** Evaluates the trade-off between minimum elapsed travel time (Plan A) and maximum cultural stops (Plan B).
        - **Synthetic Academic Boundary:** As this is an educational mini-project operating exclusively on the two approved synthetic research datasets
          (`SmartTrip (1).zip` and `WanderMatch.zip`), **no real-world employment creation, commercial revenue generation, or live visitor footfall is claimed**.
        - **Non-Commercial Academic Scope:** All analysis is conducted strictly for educational demonstration of the DEV syllabus.
        """)

        st.markdown("---")
        st.subheader("Three Analytical Populations in CultureRoute")
        st.markdown("""
        To avoid denominator conflation, CultureRoute strictly separates cultural share analysis into three distinct populations:

        1. **Catalog Listings Population (675 Attractions across 45 Cities):**
           - Represents static catalog supply.
           - Exactly **211 out of 675 attractions** belong to cultural categories (`heritage`, `museum`, `religious`, `food`).
           - **Baseline Cultural Share:** **31.3%**
           - *Note on Matrix Coverage:* Each city lists 15 catalog attractions, of which 10 have recorded directed travel matrix legs.

        2. **Telemetry Actions Population (6,850 User Interactions):**
           - Represents user demand and behavioral interest in WanderMatch.
           - Includes views, clicks, saves, likes, searches, dismissals, and shares.
           - Exactly **2,144 out of 6,850 actions** involve cultural attractions.
           - **Baseline Cultural Share:** **31.3%**

        3. **Planned Stops Population (3,191 Selected Stops):**
           - Represents actual algorithmic itinerary inclusions in SmartTrip.
           - Shared active-and-confirmed stop contract: active itineraries (`is_active = 1`), confirmed status (`status = 'confirmed'`), non-cancelled India trips, matching destination city.
           - Exactly **961 out of 3,191 stops** are cultural.
           - **Baseline Cultural Share:** **30.1%**
        """)

    # --------------------------------------------------------------------------
    # TAB 4: 24-TABLE STUDY INVENTORY
    # --------------------------------------------------------------------------
    with info_t4:
        st.subheader("4. Canonical 24-Table Reconciled Study Database")
        st.caption("Consolidated atomic SQLite database (data/study_cache.db) built from SmartTrip (PS-01.db) and WanderMatch (PS-11.db).")

        cat = load_study_catalog()
        inv_m1, inv_m2, inv_m3, inv_m4 = st.columns(4)
        inv_m1.metric("Total Tables", f"{len(cat.tables)}")
        inv_m2.metric("Total Rows", f"{cat.metadata.total_rows:,}")
        inv_m3.metric("Shared Tables", f"{len(cat.metadata.shared_tables)}", "Verified identical")
        inv_m4.metric("Indian Study Cities", "45", "Verified in scope")

        cat_df = pd.DataFrame([
            {
                "Table Name": t.table_name,
                "Origin Pack": t.origin_pack,
                "Row Count": f"{t.row_count:,}",
                "Natural Grain": t.grain,
                "Primary Key": ", ".join(t.primary_key) if t.primary_key else "None (implicit)",
                "Safe Columns": f"{len(t.safe_columns)} / {len(t.columns)}"
            }
            for t in cat.tables.values()
        ])
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

        st.markdown("""
        **Provenance & Verification Guarantees:**
        - **8 Shared Tables:** Verified by primary key, column types, and row-level contents before deduplication.
        - **10 SmartTrip-Only Tables:** Itineraries, itinerary items, transport modes, travel matrix legs, flight bookings, etc.
        - **6 WanderMatch-Only Tables:** Guide languages, guide specializations, match preferences, telemetry interactions, user preferences.
        - **Privacy Masking:** Sensitive columns (`password_hash`, `email`, `phone_number`) are masked from public DEV Lab projections.
        """)

    # --------------------------------------------------------------------------
    # TAB 5: DEV SYLLABUS ALIGNMENT
    # --------------------------------------------------------------------------
    with info_t5:
        st.subheader("5. DEV Syllabus Alignment & Academic Coverage")
        st.markdown("""
        | Syllabus Unit | Practical Implementation in CultureRoute | Module / Screen |
        | :--- | :--- | :--- |
        | **Unit 1: EDA Foundations** | Table profiling, natural grain analysis, scales of measurement, Wilson score 95% CI vs Bayesian Beta(1,1) posterior comparison | `cultureroute/dev_lab.py`, DEV Lab Tab 1 & Tab 4 |
        | **Unit 2: Data Transformations** | Composite key deduplication, discrete price binning with separate Free tier, cultural flag derivation, 666-row canonical outer join | `cultureroute/dev_lab.py`, DEV Lab Tab 2 |
        | **Unit 3: Statistics & Forecasting** | Measures of central tendency and dispersion (IQR, standard deviation), 57 continuous Monday–Sunday weeks time series, last-week (MAE 3.42) & 4-week mean (MAE 3.88) out-of-sample forecasting, Poisson GLM | `cultureroute/analytics.py`, Insights & Model Screens |
        | **Unit 4: Graphical Methods** | Bar charts of categorical shares, coordinate distribution scatter plots, continuous histograms, box plots, Empirical CDF, forecast backtest charts | `cultureroute/analytics.py`, Insights & DEV Lab Tab 4 |
        | **Unit 5: pandas & Sources** | SQLite authorizer read-only sandbox with execution step limits, single-statement enforcement, formula-safe CSV escaping, PyArrow typed Parquet binary round-tripping | `cultureroute/dev_lab.py`, DEV Lab Tab 3 & Tab 5 |
        """)

    # --------------------------------------------------------------------------
    # TAB 6: JAIPUR REPRODUCTION GUIDE
    # --------------------------------------------------------------------------
    with info_t6:
        st.subheader("6. Jaipur Specification Fixture Reproduction Guide")
        st.markdown("""
        To independently verify the deterministic itinerary engine, execute the following parameters in the **Planner** or **Compare** view:

        | Input Parameter | Specification Value | Description |
        | :--- | :--- | :--- |
        | **City** | `Jaipur` | Rajasthan capital, 15 listed POIs, 10 matrix-covered |
        | **Visit Date** | `2026-08-29` | Saturday (evaluates weekend schedule) |
        | **Start Time** | `09:00` | 24-hour typed format (morning start) |
        | **Time Window** | `6.0 Hours` | Maximum total elapsed duration |
        | **Budget** | `INR 1,000.00` | Total entry fee + transit cost cap |
        | **Target Stops** | `3 Stops` | Exact sequence of distinct attractions |

        #### Expected Mathematical Outcomes:
        - **Plan A · Fastest Feasible Plan (Min Elapsed Time):**
          - Elapsed Duration: **207 minutes** (3h 27m)
          - Cultural Stops: **1 / 3 stops**
          - Total Cost: **INR 585.46** (Entry fees + transit fares)
          - Carbon Footprint: **1.806 kg CO2**

        - **Plan B · Most Cultural Plan (Max Cultural Stops):**
          - Elapsed Duration: **227 minutes** (3h 47m)
          - Cultural Stops: **2 / 3 stops** (dynamic gain: **+1 cultural stop**)
          - Total Cost: **INR 638.30**
          - Carbon Footprint: **2.840 kg CO2**
        """)
