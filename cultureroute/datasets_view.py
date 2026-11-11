from __future__ import annotations

"""CultureRoute Datasets & Table Viewer Screen (Stage F2).

Provides:
- Data Table Viewer first: instant source selection, table choice, and paginated rows.
- Deterministic pagination with visible row ranges (e.g., "Showing rows 1–15 of 12,339").
- Transparent export: full safe exports with exact row counts (up to all 12,339 rows).
- Schema diagnostics, column data types, and null counts on demand.
- Collapsible Source Details: SmartTrip (18/32,627), WanderMatch (14/27,428), 8 shared tables,
  reconciliation to 24 canonical tables / 48,728 rows, hashes, and lineage pipelines.
- Direct seamless handoff to DEV Lab ("Open in DEV Lab →").
"""


import io
import math
from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cultureroute.ui_components import apply_editorial_chart_theme, render_section_banner

from cultureroute.study_db import (
    DEFAULT_STUDY_DB,
    load_study_catalog,
    open_study_cache,
)
from cultureroute.study_data import (
    verify_and_audit_study,
)
from cultureroute.dev_lab import (
    get_table_profile,
    export_dataframe_to_csv,
    SENSITIVE_COLUMNS,
)


@st.cache_data(ttl=3600)
def _get_datasets_summary_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    with open_study_cache() as conn:
        cat_rows = conn.execute(
            "SELECT poi_category, count(*) as count FROM activities_poi GROUP BY poi_category ORDER BY count DESC"
        ).fetchall()
        telemetry_rows = conn.execute(
            "SELECT interaction_type, count(*) as count FROM user_interactions GROUP BY interaction_type ORDER BY count DESC"
        ).fetchall()
    return pd.DataFrame([dict(r) for r in cat_rows]), pd.DataFrame([dict(r) for r in telemetry_rows])


@st.cache_data(ttl=3600)
def _export_table_csv_bytes(table_name: str, safe_cols: tuple[str, ...]) -> tuple[bytes, int]:
    with open_study_cache() as conn:
        safe_cols_sql = ", ".join([f'"{c}"' for c in safe_cols])
        export_df = pd.read_sql_query(f'SELECT {safe_cols_sql} FROM "{table_name}"', conn)
    return export_df.to_csv(index=False).encode("utf-8"), len(export_df)


@st.cache_data(ttl=3600)
def _get_paginated_table_df(
    table_name: str, safe_cols: tuple[str, ...], order_by: str, page_size: int, offset: int
) -> pd.DataFrame:
    with open_study_cache() as conn:
        safe_cols_sql = ", ".join([f'"{c}"' for c in safe_cols])
        return pd.read_sql_query(
            f'SELECT {safe_cols_sql} FROM "{table_name}" ORDER BY {order_by} LIMIT {page_size} OFFSET {offset}',
            conn,
        )


@st.cache_data(ttl=3600)
def _run_cached_df_query(sql: str) -> pd.DataFrame:
    with open_study_cache() as conn:
        return pd.read_sql_query(sql, conn)

# Hashes for the two original source packs
SMARTTRIP_HASH = "17c0a7d4bb399c4fd2b335cef5af4e02cbd21207103fea69fa9632ee6de06cc6"
WANDERMATCH_HASH = "dc8435c53e3c24ff611cfbc34c80accdc3b283c47aece49ecc52041e6eca18f8"

SHARED_TABLES = [
    "cities", "countries", "currencies", "itineraries",
    "itinerary_items", "languages", "trips", "users"
]
SMARTTRIP_ONLY = [
    "activities_poi", "airlines", "airports", "categories",
    "events_festivals", "flight_fares", "flights", "hotels",
    "place_kb", "poi_travel_matrix"
]
WANDERMATCH_ONLY = [
    "proposals", "tour_guides", "trip_members",
    "user_interactions", "user_preferences", "votes"
]


def render_datasets_screen() -> None:
    """Render the data-first Datasets Viewer screen."""
    render_section_banner(
        section_key="datasets",
        title_html="Study Datasets & <em>Table Viewer</em>",
        subtitle="Inspect SmartTrip and WanderMatch 24 canonical tables, live records, schemas, and data lineage.",
        kicker="04 / STUDY DATA CATALOG & PROVENANCE · STAGE P2",
    )
    st.markdown("Reconciled **24 Canonical Tables** and 48,728 rows from SmartTrip (1).zip and WanderMatch.zip.")

    catalog = load_study_catalog()
    all_table_names = sorted(list(catalog.tables.keys()))

    # --------------------------------------------------------------------------
    # TOP-LEVEL VISUAL DATASET ARCHITECTURE
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # TOP-LEVEL VISUAL DATASET ARCHITECTURE & STATISTICAL DISTRIBUTIONS
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background: #ffffff; border: 1.5px solid #b6afa3; border-top: 4px solid #982b32; padding: 14px 18px; margin: 16px 0 20px;">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <div style="font-family: Georgia, serif; font-size: 1.12rem; font-weight: bold; color: #292925;">
                Dataset Architecture &amp; Empirical Distributions
            </div>
            <span style="font-size: 0.74rem; font-weight: bold; color: #982b32; border: 1px solid #982b32; padding: 2px 6px;">
                48,728 VERIFIED ROWS &middot; 24 CANONICAL TABLES
            </span>
        </div>
        <div style="font-size: 0.82rem; color: #5c5852; margin-top: 4px;">
            Audited cross-pack distribution of records, table volumes, cultural categories, and user interaction telemetry from SmartTrip and WanderMatch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    v_c1, v_c2 = st.columns(2)
    with v_c1:
        origin_pie = go.Figure(data=[go.Pie(
            labels=["SmartTrip-Only (10 Tables)", "WanderMatch-Only (6 Tables)", "Shared Verified (8 Tables)"],
            values=[21300, 16101, 11327],
            hole=0.5,
            marker=dict(colors=["#982b32", "#365376", "#eee3a6"]),
            textinfo="percent+label",
            hoverinfo="label+value+percent"
        )])
        origin_pie.update_layout(
            title="48,728 Canonical Rows by Origin Pack",
            height=260,
            paper_bgcolor="#f7f3e9",
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False
        )
        apply_editorial_chart_theme(origin_pie)
        st.plotly_chart(origin_pie, use_container_width=True)

    with v_c2:
        top_tables = sorted(list(catalog.tables.values()), key=lambda t: t.row_count, reverse=True)[:8]
        top_tbl_df = pd.DataFrame([
            {"Table": t.table_name, "Rows": t.row_count, "Pack": t.origin_pack}
            for t in top_tables
        ])
        top_bar = px.bar(
            top_tbl_df,
            x="Rows",
            y="Table",
            orientation="h",
            color="Pack",
            color_discrete_map={"SmartTrip": "#982b32", "WanderMatch": "#365376", "shared": "#5c5852"},
            title="Top 8 Study Tables by Row Volume"
        )
        top_bar.update_layout(
            height=260,
            paper_bgcolor="#f7f3e9",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=35, b=10),
            yaxis=dict(autorange="reversed")
        )
        apply_editorial_chart_theme(top_bar)
        st.plotly_chart(top_bar, use_container_width=True)

    # Row 2: Cultural Category and Telemetry Distributions
    cat_df_summary, telem_df_summary = _get_datasets_summary_dfs()
    v_c3, v_c4 = st.columns(2)
    with v_c3:
        if not cat_df_summary.empty:
            cat_fig = px.pie(
                cat_df_summary,
                names="poi_category",
                values="count",
                title="Cultural POI Distribution (675 Attractions across 45 Cities)",
                color_discrete_sequence=["#982b32", "#365376", "#733339", "#4f637b", "#8b7e66", "#d4af37"],
                hole=0.45
            )
            cat_fig.update_layout(
                height=260,
                paper_bgcolor="#f7f3e9",
                margin=dict(l=10, r=10, t=35, b=10),
                showlegend=True
            )
            apply_editorial_chart_theme(cat_fig)
            st.plotly_chart(cat_fig, use_container_width=True)

    with v_c4:
        if not telem_df_summary.empty:
            telem_bar = px.bar(
                telem_df_summary,
                x="interaction_type",
                y="count",
                color="interaction_type",
                color_discrete_sequence=["#982b32", "#365376", "#733339", "#4f637b"],
                title="User Telemetry Distribution (WanderMatch 10,000+ Interactions)",
                labels={"interaction_type": "Interaction Type", "count": "Event Count"}
            )
            telem_bar.update_layout(
                height=260,
                paper_bgcolor="#f7f3e9",
                plot_bgcolor="#ffffff",
                margin=dict(l=10, r=10, t=35, b=10),
                showlegend=False
            )
            apply_editorial_chart_theme(telem_bar)
            st.plotly_chart(telem_bar, use_container_width=True)

    # --------------------------------------------------------------------------
    # CONTROLS ROW: Source Filter, Search, Table Select
    # --------------------------------------------------------------------------
    origin_choice = st.pills(
        "Filter by Origin Pack",
        ["All (24)", "SmartTrip-only (10)", "WanderMatch-only (6)", "Shared (8)"],
        default="All (24)",
        key="f2_origin_choice",
    ) or "All (24)"

    st.markdown("**Quick Table Picks:**")
    top_tbl_picks = ["trips", "activities_poi", "user_interactions", "events_festivals", "itineraries", "poi_travel_matrix"]

    def _on_datasets_pill_pick():
        picked = st.session_state.get("pills_datasets_quick_tbl")
        if picked:
            st.session_state["f2_tbl_select_val"] = picked
            st.session_state["f2_selected_table"] = picked
            st.query_params["table"] = picked

    st.pills(
        "Quick Table Picks",
        top_tbl_picks,
        key="pills_datasets_quick_tbl",
        label_visibility="collapsed",
        on_change=_on_datasets_pill_pick,
    )

    if "SmartTrip" in origin_choice:
        scoped_names = [t for t in all_table_names if t in SMARTTRIP_ONLY]
    elif "WanderMatch" in origin_choice:
        scoped_names = [t for t in all_table_names if t in WANDERMATCH_ONLY]
    elif "Shared" in origin_choice:
        scoped_names = [t for t in all_table_names if t in SHARED_TABLES]
    else:
        scoped_names = all_table_names

    c_table, c_search = st.columns([2.2, 1.8])

    with c_search:
        search_kw = st.text_input("Search Table by Name", placeholder="e.g. user_interactions", key="f2_tbl_search").strip().lower()
        if search_kw:
            scoped_names = [t for t in scoped_names if search_kw in t.lower()]

    if not scoped_names:
        st.warning("No tables match the selected source and search criteria.")
        return

    # Check query params for table handoff or default to user_interactions
    param_table = st.query_params.get("table", "")
    if param_table in scoped_names:
        default_idx = scoped_names.index(param_table)
    elif "user_interactions" in scoped_names:
        default_idx = scoped_names.index("user_interactions")
    elif "activities_poi" in scoped_names:
        default_idx = scoped_names.index("activities_poi")
    else:
        default_idx = 0

    with c_table:
        selected_tbl = st.selectbox(
            "Select Study Table to Inspect",
            scoped_names,
            index=default_idx,
            format_func=lambda x: f"{x}  [{catalog.tables[x].origin_pack}] — {catalog.tables[x].row_count:,} rows",
            key="f2_selected_table",
        )

    # --------------------------------------------------------------------------
    # TABLE PROFILE & METADATA BAR
    # --------------------------------------------------------------------------
    page_size = st.session_state.get("table_page_size", 15)
    profile = get_table_profile(selected_tbl, sample_limit=page_size)
    if not profile:
        st.error(f"Unable to load profile for table {selected_tbl}")
        return

    pack_badge_color = "#982b32" if profile.origin_pack == "SmartTrip" else ("#1e3a8a" if profile.origin_pack == "WanderMatch" else "#292925")

    # Header Card with metadata and short action controls
    st.markdown(f"""
    <div style="background: #ffffff; border: 1.5px solid var(--line); padding: 16px 20px; margin-top: 10px; margin-bottom: 14px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-family: Georgia, serif; font-size: 1.35rem; font-weight: bold; color: var(--ink);"><code>{profile.table_name}</code></span>
                <span style="margin-left: 10px; background: {pack_badge_color}; color: #ffffff; font-size: 0.72rem; padding: 2px 8px; font-weight: bold; text-transform: uppercase;">
                    {profile.origin_pack}
                </span>
            </div>
            <div style="font-size: 0.88rem; color: var(--muted);">
                Total Rows: <strong>{profile.row_count:,}</strong> &middot; Columns: <strong>{len(profile.columns)}</strong> &middot; Safe: <strong>{len(profile.safe_columns)}</strong>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 0.84rem; color: var(--ink); line-height: 1.5;">
            <strong>Natural Grain:</strong> {profile.grain} &middot;
            <strong>Primary Key:</strong> <code>{', '.join(profile.primary_key) if profile.primary_key else 'None (implicit rowid)'}</code>
            {f'&middot; <strong>Foreign Keys:</strong> {", ".join(profile.foreign_keys)}' if profile.foreign_keys else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Row: Open in DEV Lab + Download
    act_c1, act_c2, act_c3 = st.columns([1.5, 1.5, 1.0])
    with act_c1:
        st.markdown(f"""
        <a href="?view=devlab&table={profile.table_name}" target="_self" class="button" style="display: inline-block; width: 100%; text-align: center; text-decoration: none; padding: 8px 14px; font-size: 0.88rem;">
            Open in DEV Lab &rarr;
        </a>
        """, unsafe_allow_html=True)

    with act_c2:
        csv_bytes, export_row_count = _export_table_csv_bytes(
            profile.table_name, tuple(profile.safe_columns)
        )
        st.download_button(
            label=f"Download Table ({export_row_count:,} rows CSV)",
            data=csv_bytes,
            file_name=f"cultureroute_{profile.table_name}_all_{export_row_count}.csv",
            mime="text/csv",
            key=f"f2_dl_csv_{profile.table_name}",
        )

    # --------------------------------------------------------------------------
    # PAGINATED DATA RECORDS
    # --------------------------------------------------------------------------
    total_rows = profile.row_count
    total_pages = max(1, math.ceil(total_rows / page_size))
    page_key = f"f2_page_num_{profile.table_name}"
    cur_page = st.session_state.get(page_key, 1)
    if cur_page > total_pages:
        cur_page = 1

    # Fetch specific page rows deterministically from SQLite (cached)
    offset = (cur_page - 1) * page_size
    if profile.primary_key:
        order_by = ", ".join([f'"{pk}" ASC' for pk in profile.primary_key])
    else:
        order_by = "rowid ASC"

    page_df = _get_paginated_table_df(
        profile.table_name, tuple(profile.safe_columns), order_by, page_size, offset
    )

    start_row = offset + 1 if total_rows > 0 else 0
    end_row = min(offset + page_size, total_rows)

    def _go_prev_page() -> None:
        st.session_state[page_key] = max(1, cur_page - 1)

    def _go_next_page() -> None:
        st.session_state[page_key] = min(total_pages, cur_page + 1)

    tab_data, tab_schema, tab_viz = st.tabs([
        f"Data Records (Rows {start_row:,}–{end_row:,} of {total_rows:,})",
        f"Schema & Quality ({len(profile.columns)} Columns)",
        "Visual Distribution & Charts"
    ])

    with tab_data:
        # Paging Bar
        p_info, p_prev, p_curr, p_next = st.columns([2.0, 0.6, 0.8, 0.6])
        with p_info:
            st.markdown(f"**Showing rows {start_row:,} &ndash; {end_row:,} of {total_rows:,}** &middot; Page {cur_page} of {total_pages}")
        with p_prev:
            st.button(
                "Previous",
                disabled=(cur_page <= 1),
                key=f"f2_prev_{profile.table_name}",
                on_click=_go_prev_page,
            )
        with p_curr:
            st.markdown(f"<div style='text-align: center; padding-top: 6px; font-weight: bold;'>Pg {cur_page}/{total_pages}</div>", unsafe_allow_html=True)
        with p_next:
            st.button(
                "Next",
                disabled=(cur_page >= total_pages),
                key=f"f2_next_{profile.table_name}",
                on_click=_go_next_page,
            )

        st.dataframe(page_df, use_container_width=True, height=380, hide_index=True)

    with tab_schema:
        schema_records = []
        for col in profile.columns:
            null_count = profile.null_counts.get(col, 0)
            null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0
            schema_records.append({
                "Column Name": col,
                "SQL Type": profile.column_types.get(col, "TEXT"),
                "Null Count": f"{null_count:,}",
                "Null %": f"{null_pct:.1f}%",
                "Projection Status": "Permitted (Safe)" if col in profile.safe_columns else "Masked / Excluded",
            })
        st.dataframe(pd.DataFrame(schema_records), use_container_width=True, height=320, hide_index=True)

    with tab_viz:
        st.subheader(f"Visual Data Analysis: {profile.table_name}")
        st.caption(f"Interactive graphical distribution of variables for table '{profile.table_name}'.")

        if profile.table_name == "activities_poi":
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                c_df = _run_cached_df_query(
                    "SELECT poi_category, count(*) as cnt FROM activities_poi GROUP BY poi_category ORDER BY cnt DESC"
                )
                fig_c = px.pie(
                    c_df,
                    names="poi_category",
                    values="cnt",
                    hole=0.45,
                    title="Attraction Categories (675 POIs)",
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#5c5852", "#a89f91"]
                )
                fig_c.update_layout(height=320, paper_bgcolor="#f7f3e9")
                apply_editorial_chart_theme(fig_c)
                st.plotly_chart(fig_c, use_container_width=True)

            with p_c2:
                fee_df = _run_cached_df_query(
                    "SELECT CAST(entry_cost AS REAL) as fee, accessibility FROM activities_poi WHERE entry_cost IS NOT NULL"
                )
                fig_fee = px.histogram(
                    fee_df,
                    x="fee",
                    color="accessibility",
                    nbins=25,
                    title="Entry Fee Distribution by Accessibility",
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b"]
                )
                fig_fee.update_layout(height=320, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff", xaxis_title="Entry Fee (INR)", yaxis_title="POIs Count")
                apply_editorial_chart_theme(fig_fee)
                st.plotly_chart(fig_fee, use_container_width=True)

        elif profile.table_name == "user_interactions":
            u_c1, u_c2 = st.columns(2)
            with u_c1:
                i_df = _run_cached_df_query(
                    "SELECT interaction_type, count(*) as cnt FROM user_interactions GROUP BY interaction_type ORDER BY cnt DESC"
                )
                fig_int = px.pie(
                    i_df,
                    names="interaction_type",
                    values="cnt",
                    hole=0.45,
                    title="Telemetry Interactions Breakdown (12,339 Events)",
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#5c5852", "#8b7e66", "#a89f91"]
                )
                fig_int.update_layout(height=320, paper_bgcolor="#f7f3e9")
                apply_editorial_chart_theme(fig_int)
                st.plotly_chart(fig_int, use_container_width=True)

            with u_c2:
                dwell_df = _run_cached_df_query(
                    "SELECT dwell_seconds FROM user_interactions WHERE dwell_seconds > 0 AND dwell_seconds < 600"
                )
                fig_dwell = px.histogram(
                    dwell_df,
                    x="dwell_seconds",
                    nbins=30,
                    title="Dwell Duration Distribution (seconds)",
                    color_discrete_sequence=["#365376"]
                )
                fig_dwell.update_layout(height=320, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff", xaxis_title="Dwell Seconds", yaxis_title="Event Count")
                apply_editorial_chart_theme(fig_dwell)
                st.plotly_chart(fig_dwell, use_container_width=True)

        elif profile.table_name == "cities":
            cty_c1, cty_c2 = st.columns(2)
            with cty_c1:
                c_df = _run_cached_df_query(
                    "SELECT name, population FROM cities WHERE country_code='IN' ORDER BY population DESC LIMIT 15"
                )
                fig_cty = px.bar(
                    c_df,
                    x="population",
                    y="name",
                    orientation="h",
                    title="Top 15 Indian Study Cities by Population",
                    color_discrete_sequence=["#982b32"]
                )
                fig_cty.update_layout(height=340, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff", yaxis=dict(autorange="reversed"))
                apply_editorial_chart_theme(fig_cty)
                st.plotly_chart(fig_cty, use_container_width=True)

            with cty_c2:
                l_df = _run_cached_df_query(
                    "SELECT primary_language, count(*) as cnt FROM cities WHERE country_code='IN' GROUP BY primary_language ORDER BY cnt DESC"
                )
                fig_lang = px.pie(
                    l_df,
                    names="primary_language",
                    values="cnt",
                    hole=0.45,
                    title="Primary Language Distribution (45 Cities)",
                    color_discrete_sequence=["#365376", "#982b32", "#b79d5b", "#7d6f58", "#5c5852"]
                )
                fig_lang.update_layout(height=340, paper_bgcolor="#f7f3e9")
                apply_editorial_chart_theme(fig_lang)
                st.plotly_chart(fig_lang, use_container_width=True)

        elif profile.table_name == "tour_guides":
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                sp_df = _run_cached_df_query(
                    "SELECT specialisation, count(*) as cnt FROM tour_guides GROUP BY specialisation ORDER BY cnt DESC"
                )
                fig_spec = px.pie(
                    sp_df,
                    names="specialisation",
                    values="cnt",
                    hole=0.45,
                    title="Guide Domain Specialization Breakdown",
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58"]
                )
                fig_spec.update_layout(height=320, paper_bgcolor="#f7f3e9")
                apply_editorial_chart_theme(fig_spec)
                st.plotly_chart(fig_spec, use_container_width=True)

            with g_c2:
                g_num = _run_cached_df_query("SELECT rating, years_experience FROM tour_guides")
                fig_g = px.scatter(
                    g_num,
                    x="years_experience",
                    y="rating",
                    title="Guide Rating vs Experience (Years)",
                    color_discrete_sequence=["#365376"]
                )
                fig_g.update_layout(height=320, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff", xaxis_title="Years Experience", yaxis_title="Rating (1-5)")
                apply_editorial_chart_theme(fig_g)
                st.plotly_chart(fig_g, use_container_width=True)

        else:
            # Generic fallback for any other table: categorical counts
            cat_col = next((c for c in profile.columns if any(k in c.lower() for k in ["status", "category", "type", "mode", "currency"])), None)
            if cat_col:
                gen_df = _run_cached_df_query(
                    f'SELECT "{cat_col}", count(*) as cnt FROM "{profile.table_name}" GROUP BY "{cat_col}" ORDER BY cnt DESC LIMIT 10'
                )
                fig_gen = px.pie(
                    gen_df,
                    names=cat_col,
                    values="cnt",
                    hole=0.45,
                    title=f"Distribution of '{cat_col}' in {profile.table_name}",
                    color_discrete_sequence=["#982b32", "#365376", "#b79d5b", "#7d6f58", "#5c5852"]
                )
                fig_gen.update_layout(height=320, paper_bgcolor="#f7f3e9")
                apply_editorial_chart_theme(fig_gen)
                st.plotly_chart(fig_gen, use_container_width=True)
            else:
                st.info(f"Visual distribution preview active for table '{profile.table_name}'. Use DEV Lab for customized SQL aggregations.")

    # --------------------------------------------------------------------------
    # COLLAPSIBLE SOURCE DETAILS & PROVENANCE
    # --------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Source Details, Deduplication Proof & Lineage Pipelines", expanded=False):
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: var(--ink); line-height: 1.6;">
            <p><strong>Two-Pack Study Boundary & Immutable Hashes:</strong><br>
            CultureRoute operates exclusively on the two approved synthetic research packs:
            <code>SmartTrip (1).zip</code> (SHA-256: <code>{SMARTTRIP_HASH}</code>) and
            <code>WanderMatch.zip</code> (SHA-256: <code>{WANDERMATCH_HASH}</code>).
            All canonical records are stored in <code>data/study_cache.db</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid var(--line); padding: 16px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="font-family: Georgia, serif; font-weight: bold; color: #982b32;">SmartTrip (1).zip &middot; 18 Tables</div>
                <ul style="font-size: 0.82rem; margin: 8px 0; padding-left: 18px; color: var(--muted); line-height: 1.5;">
                    <li>Raw Rows: 32,627 across 18 tables</li>
                    <li>Exclusive Tables (10): activities_poi, airlines, airports, categories, events_festivals, flight_fares, flights, hotels, place_kb, poi_travel_matrix</li>
                    <li>Shared Tables (8): cities, countries, currencies, itineraries, itinerary_items, languages, trips, users</li>
                    <li>Indian POIs: 675 attractions across 45 cities (15/city)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid var(--line); padding: 16px; box-shadow: 2px 2px 0 rgba(41,41,37,0.04);">
                <div style="font-family: Georgia, serif; font-weight: bold; color: #1e3a8a;">WanderMatch.zip &middot; 14 Tables</div>
                <ul style="font-size: 0.82rem; margin: 8px 0; padding-left: 18px; color: var(--muted); line-height: 1.5;">
                    <li>Raw Rows: 27,428 across 14 tables</li>
                    <li>Exclusive Tables (6): proposals, tour_guides, trip_members, user_interactions, user_preferences, votes</li>
                    <li>Shared Tables (8): Verified identical keys & content</li>
                    <li>Interaction Events: 12,339 telemetry records</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #fdfbf7; border: 1px solid var(--line); padding: 16px; margin-top: 14px;">
            <div style="font-family: Georgia, serif; font-weight: bold; color: var(--ink);">
                Canonical Reconciliation Formula
            </div>
            <div style="font-size: 0.84rem; color: var(--muted); margin-top: 6px; line-height: 1.5;">
                <strong>32,627 (SmartTrip) + 27,428 (WanderMatch) &minus; 11,327 (Deduplicated Shared Rows) = 48,728 Canonical Rows.</strong><br>
                The 8 shared tables match identically on primary keys, column types, and row contents.
                Duplicates are eliminated at database compilation time so each record exists exactly once in the study cache.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 14px; font-size: 0.84rem; line-height: 1.6;">
            <strong>Data Lineage Walkthroughs:</strong>
            <ul style="margin: 6px 0; padding-left: 20px;">
                <li><strong>Lineage 1 (Interaction Telemetry):</strong> <code>user_interactions</code> (12,339 rows) &rarr; match to <code>activities_poi</code> on <code>entity_id = poi_id</code> &rarr; scope to Indian cities (6,850 actions) &rarr; filter cultural categories (2,144 actions) &rarr; feeds <strong>Insights time series</strong> and <strong>Prediction Model panel</strong>.</li>
                <li><strong>Lineage 2 (Itinerary Stops):</strong> <code>trips</code> &rarr; <code>itineraries</code> &rarr; <code>itinerary_items</code> (8,583 scheduled stops) &rarr; filter non-cancelled Indian study trips (411 trips, 3,191 stops) &rarr; filter cultural stops (961 stops) &rarr; feeds <strong>Planner Day Routing</strong> and <strong>DEV Lab 352/11/303 Join Partition</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
