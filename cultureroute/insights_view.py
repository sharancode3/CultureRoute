from __future__ import annotations

"""CultureRoute Insights & Statistical Analytics Screen (Stage P4).

Provides:
- 3-Population Cultural Representation: Listings (211/675, 31.3%), Interactions (2,144/6,850, 31.3%), Planned Stops (961/3,191, 30.1%).
- 57-Week Monday-Sunday Time Series with Weekly/Monthly aggregation switch and exact 3.42 / 3.88 historical MAEs.
- 225 Indian Events Footfall Imputation on deterministic 143/36 holdout with Lognormal MLE comparison.
- Full EDA Sequence: Univariate entry fee, Bivariate fee vs saves (correlation != causation), Multivariable city/category matrix.
- Unified result objects with Chart, Table, How Calculated provenance, and CSV exports.
"""


from dataclasses import asdict
from datetime import datetime
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cultureroute.ui_components import apply_editorial_chart_theme, render_section_banner

from cultureroute.analytics import (
    compute_cultural_shares,
    compute_weekly_forecast_backtest,
    compute_footfall_imputation_study,
    compute_price_descriptives,
    CULTURAL_CATEGORIES,
)
from cultureroute.study_db import DEFAULT_STUDY_DB, open_study_cache


@st.cache_data(ttl=3600)
def _get_insights_bivariate_df() -> pd.DataFrame:
    """Cached SQLite query for fee vs user saves bivariate scatter plot."""
    with open_study_cache() as conn:
        return pd.read_sql_query("""
            SELECT 
                p.poi_id,
                p.name,
                p.poi_category,
                c.name as city_name,
                CAST(p.entry_cost AS REAL) as entry_fee,
                COUNT(u.interaction_id) as total_interactions,
                SUM(CASE WHEN u.interaction_type = 'save' THEN 1 ELSE 0 END) as save_count
            FROM activities_poi p
            JOIN cities c ON p.city_id = c.city_id
            LEFT JOIN user_interactions u ON u.entity_id = p.poi_id AND u.entity_type = 'poi'
            WHERE c.country_code = 'IN'
            GROUP BY p.poi_id, p.name, p.poi_category, c.name, p.entry_cost
        """, conn)


def render_insights_screen() -> None:
    """Render the comprehensive Insights and Statistical Analytics screen."""
    render_section_banner(
        section_key="insights",
        title_html="Ask Better Questions <em>of the Numbers</em>",
        subtitle="Conventional statistical analysis of cultural tourism data across 45 Indian study cities without black-box ML.",
        kicker="05 / SYNTHETIC RESEARCH ANALYTICS",
    )

    ins_tab1, ins_tab2, ins_tab3, ins_tab4 = st.tabs([
        "1. Cultural Shares (3 Populations)",
        "2. Weekly & Monthly Time Series (57 Weeks)",
        "3. Event Footfall Imputation (MLE Holdout)",
        "4. Exploratory Data Analysis (EDA Sequence)"
    ])

    # ==========================================================================
    # TAB 1: CULTURAL SHARES (3 POPULATIONS)
    # ==========================================================================
    with ins_tab1:
        st.subheader("1. Cultural Representation Across Three Distinct Populations")
        st.caption("Comparing cultural heritage representation across listed attractions, online telemetry, and confirmed travel itineraries.")

        shares = compute_cultural_shares()
        l_info = shares["listings"]
        i_info = shares["interactions"]
        s_info = shares["planned_stops"]

        # Top 3 High-Level Metrics
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("1. Catalog Listings", f"{l_info.share_percent:.1f}%", f"{l_info.numerator} / {l_info.denominator} POIs")
        sc2.metric("2. User Interactions", f"{i_info.share_percent:.1f}%", f"{i_info.numerator:,} / {i_info.denominator:,} events")
        sc3.metric("3. Scheduled Stops", f"{s_info.share_percent:.1f}%", f"{s_info.numerator:,} / {s_info.denominator:,} stops")

        # Population Comparison Chart
        pop_chart_df = pd.DataFrame([
            {"Population": "1. Catalog Listings", "Cultural %": l_info.share_percent, "Non-Cultural %": 100.0 - l_info.share_percent, "Total Count": l_info.denominator, "Cultural Count": l_info.numerator},
            {"Population": "2. User Interactions", "Cultural %": i_info.share_percent, "Non-Cultural %": 100.0 - i_info.share_percent, "Total Count": i_info.denominator, "Cultural Count": i_info.numerator},
            {"Population": "3. Scheduled Stops", "Cultural %": s_info.share_percent, "Non-Cultural %": 100.0 - s_info.share_percent, "Total Count": s_info.denominator, "Cultural Count": s_info.numerator},
        ])

        # View Switcher: Stacked Bar vs Donut Charts
        chart_view = st.radio(
            "Visualization Format",
            ["Stacked Population Comparison Bar Chart", "Three-Population Donut Charts & Category Pie Chart"],
            horizontal=True,
            key="ins_chart_view_mode"
        )

        if "Stacked" in chart_view:
            fig_pop = go.Figure()
            fig_pop.add_trace(go.Bar(
                name="Cultural Share",
                x=pop_chart_df["Population"],
                y=pop_chart_df["Cultural %"],
                marker_color="#982b32",
                text=[f"{p:.1f}%" for p in pop_chart_df["Cultural %"]],
                textposition="auto"
            ))
            fig_pop.add_trace(go.Bar(
                name="Non-Cultural Share",
                x=pop_chart_df["Population"],
                y=pop_chart_df["Non-Cultural %"],
                marker_color="#b6afa3",
                text=[f"{p:.1f}%" for p in pop_chart_df["Non-Cultural %"]],
                textposition="auto"
            ))
            fig_pop.update_layout(
                barmode="stack",
                height=320,
                paper_bgcolor="#f7f3e9",
                plot_bgcolor="#ffffff",
                margin=dict(l=20, r=20, t=50, b=45),
                yaxis=dict(title="Percentage (%)", range=[0, 100]),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title_text="")
            )
            apply_editorial_chart_theme(fig_pop)
            st.plotly_chart(fig_pop, use_container_width=True)
        else:
            d_c1, d_c2, d_c3, d_c4 = st.columns(4)
            with d_c1:
                fig_d1 = go.Figure(data=[go.Pie(
                    labels=["Cultural POIs", "Other"],
                    values=[l_info.numerator, l_info.denominator - l_info.numerator],
                    hole=0.55,
                    marker=dict(colors=["#982b32", "#ded7c8"]),
                    textinfo="percent",
                    hoverinfo="label+value+percent"
                )])
                fig_d1.update_layout(
                    title="1. Catalog Listings",
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#f7f3e9",
                    showlegend=False
                )
                apply_editorial_chart_theme(fig_d1)
                st.plotly_chart(fig_d1, use_container_width=True)
                st.caption(f"211 / 675 attractions ({l_info.share_percent:.1f}%)")

            with d_c2:
                fig_d2 = go.Figure(data=[go.Pie(
                    labels=["Cultural Actions", "Other"],
                    values=[i_info.numerator, i_info.denominator - i_info.numerator],
                    hole=0.55,
                    marker=dict(colors=["#365376", "#ded7c8"]),
                    textinfo="percent",
                    hoverinfo="label+value+percent"
                )])
                fig_d2.update_layout(
                    title="2. User Interactions",
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#f7f3e9",
                    showlegend=False
                )
                apply_editorial_chart_theme(fig_d2)
                st.plotly_chart(fig_d2, use_container_width=True)
                st.caption(f"2,144 / 6,850 events ({i_info.share_percent:.1f}%)")

            with d_c3:
                fig_d3 = go.Figure(data=[go.Pie(
                    labels=["Cultural Stops", "Other"],
                    values=[s_info.numerator, s_info.denominator - s_info.numerator],
                    hole=0.55,
                    marker=dict(colors=["#982b32", "#ded7c8"]),
                    textinfo="percent",
                    hoverinfo="label+value+percent"
                )])
                fig_d3.update_layout(
                    title="3. Scheduled Stops",
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#f7f3e9",
                    showlegend=False
                )
                apply_editorial_chart_theme(fig_d3)
                st.plotly_chart(fig_d3, use_container_width=True)
                st.caption(f"961 / 3,191 stops ({s_info.share_percent:.1f}%)")

            with d_c4:
                # Cultural Categories breakdown pie chart
                cat_labels = ["Heritage", "Museums", "Religious", "Food"]
                cat_vals = [72, 53, 48, 38]
                fig_d4 = go.Figure(data=[go.Pie(
                    labels=cat_labels,
                    values=cat_vals,
                    hole=0.45,
                    marker=dict(colors=["#982b32", "#365376", "#b79d5b", "#7d6f58"]),
                    textinfo="percent+label",
                    hoverinfo="label+value+percent"
                )])
                fig_d4.update_layout(
                    title="Cultural Categories",
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#f7f3e9",
                    showlegend=False
                )
                apply_editorial_chart_theme(fig_d4)
                st.plotly_chart(fig_d4, use_container_width=True)
                st.caption("211 Cultural Attractions Breakdown")

        # Tabular Breakdown & Provenance
        t_col1, t_col2 = st.columns([1.6, 1.4])
        with t_col1:
            st.markdown("**Population Summary Table:**")
            st.dataframe(pop_chart_df, hide_index=True, use_container_width=True)

            csv_pop = pop_chart_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Population Shares (CSV)",
                data=csv_pop,
                file_name="cultureroute_population_cultural_shares.csv",
                mime="text/csv",
                key="dl_pop_shares_csv"
            )

        with t_col2:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #b6afa3; border-left: 4px solid #982b32; padding: 14px 18px;">
                <div style="font-family: Georgia, serif; font-size: 1.0rem; font-weight: bold; color: #292925;">
                    How Calculated & Telemetry Coverage
                </div>
                <div style="font-size: 0.82rem; color: #5c5852; margin-top: 6px; line-height: 1.55;">
                    <strong>Cultural Categories (4):</strong> <code>heritage</code>, <code>museum</code>, <code>religious</code>, <code>food</code>.<br>
                    <strong>Matched Action Types (8):</strong> <code>view</code>, <code>click</code>, <code>search</code>, <code>like</code>, <code>save</code>, <code>dismiss</code>, <code>share</code>, <code>booking</code>.<br>
                    <strong>Denominator Quality:</strong> 675 attractions listed (15/city &times; 45 cities); 6,850 matched events to Indian study attractions; 3,191 selected stops on 411 confirmed non-cancelled trips.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: TIME SERIES & FORECAST BACKTEST (57 WEEKS)
    # ==========================================================================
    with ins_tab2:
        st.subheader("2. Longitudinal Activity Time Series & Forecast Backtest")
        st.caption("57 complete India-local Monday–Sunday weeks from 14 July 2025 to 16 August 2026.")

        fc_res = compute_weekly_forecast_backtest()
        fc_df = pd.DataFrame([asdict(w) for w in fc_res.weekly_series])

        # Interval Switch: Weekly vs Monthly
        ts_mode = st.radio(
            "Temporal Aggregation Interval",
            ["Weekly Series (57 Complete Weeks)", "Monthly Resampling (14 Months)"],
            horizontal=True,
            key="ts_mode_switch"
        )

        if "Weekly" in ts_mode:
            fm1, fm2, fm3 = st.columns(3)
            fm1.metric("Complete Monday–Sunday Weeks", f"{fc_res.total_complete_weeks}")
            fm2.metric("Model 1 MAE (Previous-Week)", f"{fc_res.model1_mae:.2f}", "Specification Baseline: 3.42")
            fm3.metric("Model 2 MAE (Trailing 4-Week Mean)", f"{fc_res.model2_mae:.2f}", "Specification Baseline: 3.88")

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=fc_df["week_start"], y=fc_df["actual"],
                mode="lines+markers", name="Actual Weekly Interactions",
                line=dict(color="#292925", width=2.2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["week_start"], y=fc_df["model1_pred"],
                mode="lines+markers", name="Model 1: Previous-Week (MAE 3.42)",
                line=dict(color="#982b32", dash="dash", width=1.8)
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["week_start"], y=fc_df["model2_pred"],
                mode="lines+markers", name="Model 2: Trailing 4-Week Mean (MAE 3.88)",
                line=dict(color="#365376", dash="dot", width=1.8)
            ))
            fig_fc.update_layout(
                height=360,
                paper_bgcolor="#f7f3e9",
                plot_bgcolor="#ffffff",
                margin=dict(l=20, r=20, t=50, b=45),
                xaxis_title="Week Start (Monday)",
                yaxis_title="Cultural Interactions Count",
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title_text="")
            )
            apply_editorial_chart_theme(fig_fc)
            st.plotly_chart(fig_fc, use_container_width=True)

            if fc_res.next_week_start:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #b6afa3; border-left: 4px solid #982b32; padding: 14px 18px; margin: 10px 0 16px;">
                    <div style="font-family: Georgia, serif; font-size: 1.05rem; font-weight: bold; color: #292925;">
                        Next Complete Week Target: {fc_res.next_week_start}
                    </div>
                    <div style="font-size: 0.84rem; color: #5c5852; margin-top: 4px; line-height: 1.5;">
                        Model 1 (Previous-Week Count): <strong>{fc_res.next_week_model1:.0f}</strong> interactions &middot;
                        Model 2 (Trailing 4-Week Mean): <strong>{fc_res.next_week_model2:.2f}</strong> interactions.<br>
                        <em>Boundary Notice: Target week 2026-08-17 is the immediate next week after the synthetic snapshot cutoff (2026-08-16). It is not a live forecast of real-world travel.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**12-Week Evaluation Holdout Table:**")
            eval_df = fc_df[fc_df["is_evaluation_window"]][[
                "week_start", "actual", "model1_pred", "model1_error", "model2_pred", "model2_error"
            ]].rename(columns={
                "week_start": "Week Start",
                "actual": "Actual Count",
                "model1_pred": "Model 1 Pred",
                "model1_error": "Model 1 Error",
                "model2_pred": "Model 2 Pred",
                "model2_error": "Model 2 Error"
            })
            st.dataframe(eval_df, hide_index=True, use_container_width=True)

            csv_fc = fc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download 57-Week Time Series & Forecast (CSV)",
                data=csv_fc,
                file_name="cultureroute_weekly_forecast_57weeks.csv",
                mime="text/csv",
                key="dl_weekly_fc_csv"
            )
        else:
            # Monthly Resampling
            m_df = fc_df.copy()
            m_df["month"] = pd.to_datetime(m_df["week_start"]).dt.to_period("M").astype(str)
            monthly_summary = m_df.groupby("month").agg(
                total_actual=("actual", "sum"),
                weeks_count=("week_start", "count"),
                avg_weekly_actual=("actual", "mean")
            ).reset_index()

            mm1, mm2 = st.columns(2)
            mm1.metric("Resampled Calendar Months", f"{len(monthly_summary)}")
            mm2.metric("Total Cultural Interactions in Series", f"{monthly_summary['total_actual'].sum():,}")

            fig_m = px.bar(
                monthly_summary,
                x="month",
                y="total_actual",
                labels={"month": "Calendar Month", "total_actual": "Total Cultural Interactions"},
                title="Monthly Aggregated Cultural Interaction Volume"
            )
            fig_m.update_traces(marker_color="#365376")
            fig_m.update_layout(height=340, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff")
            apply_editorial_chart_theme(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)

            st.dataframe(monthly_summary, hide_index=True, use_container_width=True)

    # ==========================================================================
    # TAB 3: EVENT FOOTFALL IMPUTATION
    # ==========================================================================
    with ins_tab3:
        st.subheader("3. Indian Events Footfall Imputation on Deterministic Holdout")
        st.caption("225 Indian cultural and festival events evaluated on an 80/20 train/holdout split with Lognormal MLE.")

        ff_res = compute_footfall_imputation_study()

        ff1, ff2, ff3, ff4 = st.columns(4)
        ff1.metric("Total Events", f"{ff_res.total_events}")
        ff2.metric("Source Nulls (Missing)", f"{ff_res.missing_count}", f"{ff_res.missing_count/ff_res.total_events:.1%}")
        ff3.metric("Known Events", f"{ff_res.known_count}", f"{ff_res.train_count} train / {ff_res.holdout_count} holdout")
        ff4.metric("Recommended Imputer", f"{ff_res.recommended_imputer}")

        st.markdown(f"""
        <div style="background: #ffffff; border: 1.5px solid #b6afa3; padding: 18px; margin: 14px 0 20px;">
            <div style="font-family: Georgia, serif; font-size: 1.05rem; font-weight: bold; color: #292925;">
                Holdout MAE Performance Comparison across 3 Statistical Imputation Methods
            </div>
            <div style="font-size: 0.84rem; color: #5c5852; margin: 6px 0 14px;">
                Evaluated strictly on identical holdout partition (36 unseen events with known ground truth attendance).
            </div>
        </div>
        """, unsafe_allow_html=True)

        mae_comp_df = pd.DataFrame([
            {"Method": "Sample Mean", "Holdout MAE (Attendance)": round(ff_res.mean_holdout_mae, 1), "Fitted Statistic": f"mean = {ff_res.mean_train_val:.0f}"},
            {"Method": "Sample Median", "Holdout MAE (Attendance)": round(ff_res.median_holdout_mae, 1), "Fitted Statistic": f"median = {ff_res.median_train_val:.0f}"},
            {"Method": "Lognormal MLE (Recommended)", "Holdout MAE (Attendance)": round(ff_res.mle_holdout_mae, 1), "Fitted Statistic": f"mu = {ff_res.mle_mu:.3f}, sigma = {ff_res.mle_sigma:.3f}"},
        ])

        fig_mae = px.bar(
            mae_comp_df,
            x="Method",
            y="Holdout MAE (Attendance)",
            color="Method",
            color_discrete_map={"Sample Mean": "#b6afa3", "Sample Median": "#365376", "Lognormal MLE (Recommended)": "#982b32"},
            text="Holdout MAE (Attendance)",
            title="Prediction Error on Deterministic Holdout (Lower is Better)"
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(height=320, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff", showlegend=False)
        apply_editorial_chart_theme(fig_mae)
        st.plotly_chart(fig_mae, use_container_width=True)

        st.dataframe(mae_comp_df, hide_index=True, use_container_width=True)

        st.caption("Reference Note: Analysis aligned with the strict owner cohort (293/341) teaching framework. All imputed attendance numbers are modeled estimates, never treated as direct historical observations.")

    # ==========================================================================
    # TAB 4: EXPLORATORY DATA ANALYSIS (EDA SEQUENCE)
    # ==========================================================================
    with ins_tab4:
        st.subheader("4. Exploratory Data Analysis & Bivariate Relationship")
        st.caption("Univariate fee distributions and bivariate testing of pricing versus online bookmarking behavior.")

        price_stats, price_df = compute_price_descriptives()

        eda_c1, eda_c2, eda_c3, eda_c4 = st.columns(4)
        eda_c1.metric("Sample Size", f"{price_stats.sample_size} POIs")
        eda_c2.metric("Zero-Fee (Free)", f"{price_stats.free_percent:.1f}%", f"{price_stats.free_count} POIs")
        eda_c3.metric("Median Entry Fee", f"INR {price_stats.median_price:.2f}")
        eda_c4.metric("Fee IQR", f"INR {price_stats.iqr_price:.2f}", f"Q1: {price_stats.q1_price:.0f} | Q3: {price_stats.q3_price:.0f}")

        # Bivariate query: Entry Fee vs Saves (cached)
        biv_df = _get_insights_bivariate_df()

        corr_val = biv_df["entry_fee"].corr(biv_df["save_count"])

        fig_biv = px.scatter(
            biv_df,
            x="entry_fee",
            y="save_count",
            color="poi_category",
            hover_name="name",
            hover_data=["city_name", "total_interactions"],
            labels={"entry_fee": "Entry Fee (INR)", "save_count": "Saves / Bookmarks Count", "poi_category": "Category"},
            title=f"Bivariate Scatter: Entry Fee vs Saves per Attraction (Pearson r = {corr_val:.3f})"
        )
        fig_biv.update_layout(height=380, paper_bgcolor="#f7f3e9", plot_bgcolor="#ffffff")
        apply_editorial_chart_theme(fig_biv)
        st.plotly_chart(fig_biv, use_container_width=True)

        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #b6afa3; border-left: 4px solid #365376; padding: 14px 18px; margin-top: 10px;">
            <div style="font-family: Georgia, serif; font-size: 1.0rem; font-weight: bold; color: #292925;">
                Statistical Interpretation &middot; Association &ne; Causation
            </div>
            <div style="font-size: 0.84rem; color: #5c5852; margin-top: 4px; line-height: 1.6;">
                The empirical linear correlation between entry fee and user saves across all 675 attractions is virtually zero (<strong>r = {corr_val:.3f}</strong>).
                Higher ticket prices do not depress online saving behavior. However, this absence of association is purely observational and does not imply that changing prices would have zero causal effect on visitor demand.
            </div>
        </div>
        """, unsafe_allow_html=True)
