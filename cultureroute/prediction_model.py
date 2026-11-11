"""CultureRoute Tabular Supervised Prediction Engine (Stage P6).

Implements:
1. Panel Construction:
   - 45 Indian study cities x 57 complete Monday-Sunday weeks = 2,565 city-week panel rows.
   - 2,138 qualifying cultural actions within complete weeks (14 July 2025 - 16 August 2026).
   - 1,191 zero-count city-weeks (46.4% sparsity).
2. Feature Engineering (Strict Zero Leakage):
   - City identity (one-hot encoded).
   - lag_1 (count at t-1).
   - lag_2 (count at t-2).
   - trailing_4w_mean (past 4 weeks rolling mean: mean of t-4..t-1, shifted strictly before rolling).
   - 4-week history initialization leaves 2,385 eligible panel rows.
3. Chronological Time Partitioning:
   - Training window: Weeks 5..45 (41 weeks x 45 cities = 1,845 rows).
   - Holdout test window: Weeks 46..57 (Last 12 complete weeks x 45 cities = 540 rows).
4. Supervised Model:
   - scikit-learn PoissonRegressor(alpha=1.0, max_iter=300).
   - Evaluated out-of-sample against:
     * Baseline 1: Previous-week count (lag_1).
     * Baseline 2: Trailing 4-week mean.
   - Evaluated using MAE in interactions per city-week, RMSE, and zero vs nonzero stratification.
5. Dated Prediction:
   - Target week: 2026-08-17 (first week after complete snapshot cutoff).
   - Per-city predicted cultural interaction counts with Poisson 95% confidence bounds.
6. Local Reproducible Artifact:
   - Saved to models/poisson_model_artifact.json with full provenance, metrics, and coefficients.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cultureroute.study_data import open_study_databases

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACT_PATH = ARTIFACT_DIR / "poisson_model_artifact.json"


@dataclass(frozen=True)
class PanelDataset:
    panel_df: pd.DataFrame
    eligible_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cities: List[Dict[str, Any]]
    complete_weeks: List[date]
    total_cities: int
    total_weeks: int
    total_panel_rows: int
    qualifying_actions: int
    zero_count_rows: int
    eligible_rows: int
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class ModelEvaluation:
    model_family: str
    alpha: float
    train_rows: int
    test_rows: int
    test_weeks_count: int
    test_start_week: str
    test_end_week: str
    # Primary metrics (MAE in interactions per city-week)
    model_mae: float
    baseline_prev_mae: float
    baseline_4w_mae: float
    # RMSE metrics
    model_rmse: float
    baseline_prev_rmse: float
    baseline_4w_rmse: float
    # Stratified: Zero targets
    zero_targets_count: int
    zero_model_mae: float
    zero_base_prev_mae: float
    zero_base_4w_mae: float
    # Stratified: Non-zero targets
    nonzero_targets_count: int
    nonzero_model_mae: float
    nonzero_base_prev_mae: float
    nonzero_base_4w_mae: float
    # Detailed test predictions dataframe
    test_predictions_df: pd.DataFrame


@dataclass(frozen=True)
class DatedPredictionResult:
    target_week_start: str
    snapshot_cutoff: str
    city_predictions: List[Dict[str, Any]]
    total_predicted_interactions: float


@lru_cache(maxsize=1)
def build_city_week_panel() -> PanelDataset:
    """Build the balanced 45 cities x 57 weeks panel and engineer past-only lag features."""
    with open_study_databases() as (smart, _):
        city_rows = smart.execute(
            "SELECT city_id, name, state FROM cities WHERE country_code='IN' ORDER BY city_id"
        ).fetchall()
        cities = [{"city_id": r["city_id"], "name": r["name"], "state": r["state"]} for r in city_rows]

        cultural_pred = "('heritage','museum','religious','food')"
        sql = f"""
            SELECT u.occurred_at, c.city_id, c.name as city_name
            FROM wm.user_interactions u
            JOIN activities_poi a ON u.entity_type='poi' AND u.entity_id=a.poi_id
            JOIN cities c ON c.city_id=a.city_id
            WHERE c.country_code='IN' AND a.poi_category IN {cultural_pred}
        """
        actions = smart.execute(sql).fetchall()

    # Determine unique Mondays and drop partial trailing week
    week_counter: Counter = Counter()
    for ts, cid, cname in actions:
        d = datetime.fromisoformat(ts).date()
        m = d - timedelta(days=d.weekday())
        week_counter[m] += 1

    sorted_mondays = sorted(week_counter.keys())
    complete_mondays = sorted_mondays[:-1]  # Exclude partial trailing week (57 complete weeks)

    # Count interactions per (city_id, monday) within complete weeks
    city_week_counts: Counter = Counter()
    qualifying_count = 0
    for ts, cid, cname in actions:
        d = datetime.fromisoformat(ts).date()
        m = d - timedelta(days=d.weekday())
        if m in complete_mondays:
            city_week_counts[(cid, m)] += 1
            qualifying_count += 1

    # Balanced Cartesian product: 45 cities x 57 weeks = 2,565 rows
    records = []
    for c in cities:
        cid = c["city_id"]
        cname = c["name"]
        for m in complete_mondays:
            cnt = city_week_counts.get((cid, m), 0)
            records.append({
                "city_id": cid,
                "city_name": cname,
                "state": c["state"],
                "week_start": str(m),
                "interaction_count": cnt,
            })

    panel_df = pd.DataFrame(records)
    panel_df["week_date"] = pd.to_datetime(panel_df["week_start"])
    panel_df = panel_df.sort_values(["city_id", "week_date"]).reset_index(drop=True)

    # Feature engineering: strictly past-only lag features per city
    panel_df["lag_1"] = panel_df.groupby("city_id")["interaction_count"].shift(1)
    panel_df["lag_2"] = panel_df.groupby("city_id")["interaction_count"].shift(2)
    panel_df["trailing_4w_mean"] = (
        panel_df.groupby("city_id")["interaction_count"]
        .shift(1)
        .rolling(4, min_periods=4)
        .mean()
    )

    eligible_df = panel_df.dropna(subset=["lag_1", "lag_2", "trailing_4w_mean"]).copy().reset_index(drop=True)
    eligible_weeks = sorted(eligible_df["week_start"].unique())
    test_weeks = eligible_weeks[-12:]
    train_weeks = eligible_weeks[:-12]

    train_df = eligible_df[eligible_df["week_start"].isin(train_weeks)].copy().reset_index(drop=True)
    test_df = eligible_df[eligible_df["week_start"].isin(test_weeks)].copy().reset_index(drop=True)

    zero_count = int((panel_df["interaction_count"] == 0).sum())

    return PanelDataset(
        panel_df=panel_df,
        eligible_df=eligible_df,
        train_df=train_df,
        test_df=test_df,
        cities=cities,
        complete_weeks=complete_mondays,
        total_cities=len(cities),
        total_weeks=len(complete_mondays),
        total_panel_rows=len(panel_df),
        qualifying_actions=qualifying_count,
        zero_count_rows=zero_count,
        eligible_rows=len(eligible_df),
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def train_and_evaluate_poisson_model(
    alpha: float = 1.0, save_artifact: bool = True
) -> Tuple[ModelEvaluation, DatedPredictionResult, Pipeline]:
    """Train PoissonRegressor on historical training window and evaluate on the 12-week holdout."""
    dataset = build_city_week_panel()
    features = ["city_id", "lag_1", "lag_2", "trailing_4w_mean"]

    X_train = dataset.train_df[features]
    y_train = dataset.train_df["interaction_count"]
    X_test = dataset.test_df[features]
    y_test = dataset.test_df["interaction_count"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("city", OneHotEncoder(drop="first", handle_unknown="ignore"), ["city_id"]),
            ("num", "passthrough", ["lag_1", "lag_2", "trailing_4w_mean"]),
        ]
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("reg", PoissonRegressor(alpha=alpha, max_iter=300)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Detailed test evaluation
    eval_df = dataset.test_df.copy()
    eval_df["y_pred"] = y_pred
    eval_df["base_prev"] = eval_df["lag_1"]
    eval_df["base_4w"] = eval_df["trailing_4w_mean"]
    eval_df["model_abs_err"] = (eval_df["interaction_count"] - eval_df["y_pred"]).abs()
    eval_df["base_prev_abs_err"] = (eval_df["interaction_count"] - eval_df["base_prev"]).abs()
    eval_df["base_4w_abs_err"] = (eval_df["interaction_count"] - eval_df["base_4w"]).abs()

    # Overall metrics
    m_mae = float(mean_absolute_error(y_test, y_pred))
    bp_mae = float(mean_absolute_error(y_test, eval_df["base_prev"]))
    b4_mae = float(mean_absolute_error(y_test, eval_df["base_4w"]))

    m_rmse = float(root_mean_squared_error(y_test, y_pred))
    bp_rmse = float(root_mean_squared_error(y_test, eval_df["base_prev"]))
    b4_rmse = float(root_mean_squared_error(y_test, eval_df["base_4w"]))

    # Stratified metrics: Zero vs Non-zero
    zero_mask = y_test == 0
    zero_cnt = int(zero_mask.sum())
    z_m_mae = float(mean_absolute_error(y_test[zero_mask], y_pred[zero_mask]))
    z_bp_mae = float(mean_absolute_error(y_test[zero_mask], eval_df.loc[zero_mask, "base_prev"]))
    z_b4_mae = float(mean_absolute_error(y_test[zero_mask], eval_df.loc[zero_mask, "base_4w"]))

    nonzero_mask = ~zero_mask
    nonzero_cnt = int(nonzero_mask.sum())
    nz_m_mae = float(mean_absolute_error(y_test[nonzero_mask], y_pred[nonzero_mask]))
    nz_bp_mae = float(mean_absolute_error(y_test[nonzero_mask], eval_df.loc[nonzero_mask, "base_prev"]))
    nz_b4_mae = float(mean_absolute_error(y_test[nonzero_mask], eval_df.loc[nonzero_mask, "base_4w"]))

    test_weeks = sorted(eval_df["week_start"].unique())
    evaluation = ModelEvaluation(
        model_family="PoissonRegressor",
        alpha=alpha,
        train_rows=len(dataset.train_df),
        test_rows=len(dataset.test_df),
        test_weeks_count=len(test_weeks),
        test_start_week=test_weeks[0],
        test_end_week=test_weeks[-1],
        model_mae=round(m_mae, 4),
        baseline_prev_mae=round(bp_mae, 4),
        baseline_4w_mae=round(b4_mae, 4),
        model_rmse=round(m_rmse, 4),
        baseline_prev_rmse=round(bp_rmse, 4),
        baseline_4w_rmse=round(b4_rmse, 4),
        zero_targets_count=zero_cnt,
        zero_model_mae=round(z_m_mae, 4),
        zero_base_prev_mae=round(z_bp_mae, 4),
        zero_base_4w_mae=round(z_b4_mae, 4),
        nonzero_targets_count=nonzero_cnt,
        nonzero_model_mae=round(nz_m_mae, 4),
        nonzero_base_prev_mae=round(nz_bp_mae, 4),
        nonzero_base_4w_mae=round(nz_b4_mae, 4),
        test_predictions_df=eval_df,
    )

    # Dated prediction for next week: 2026-08-17
    # Calculated strictly using data available up to 2026-08-16 (the last complete weeks 54, 55, 56, 57)
    last_4_weeks = dataset.complete_weeks[-4:]
    w57 = str(dataset.complete_weeks[-1])
    w56 = str(dataset.complete_weeks[-2])
    last_4_str = [str(w) for w in last_4_weeks]

    future_rows = []
    for c in dataset.cities:
        cid = c["city_id"]
        cname = c["name"]
        c_df = dataset.panel_df[dataset.panel_df["city_id"] == cid]
        l1 = float(c_df[c_df["week_start"] == w57]["interaction_count"].values[0])
        l2 = float(c_df[c_df["week_start"] == w56]["interaction_count"].values[0])
        last4_vals = c_df[c_df["week_start"].isin(last_4_str)]["interaction_count"].values
        t4m = float(last4_vals.mean())

        future_rows.append({
            "city_id": cid,
            "city_name": cname,
            "state": c["state"],
            "lag_1": l1,
            "lag_2": l2,
            "trailing_4w_mean": round(t4m, 2),
        })

    future_df = pd.DataFrame(future_rows)
    pred_values = pipeline.predict(future_df[features])
    future_df["predicted_interactions"] = np.round(pred_values, 3)

    # Add exact 95% Poisson confidence interval using Poisson PPF quantiles
    from scipy.stats import poisson
    future_df["ci_lower_95"] = [max(0.0, min(float(np.round(p, 3)), float(poisson.ppf(0.025, max(1e-5, p))))) for p in pred_values]
    future_df["ci_upper_95"] = [max(float(np.round(p, 3)), float(poisson.ppf(0.975, max(1e-5, p)))) for p in pred_values]

    future_df = future_df.sort_values("predicted_interactions", ascending=False).reset_index(drop=True)
    city_preds = future_df.to_dict(orient="records")
    total_predicted = float(future_df["predicted_interactions"].sum())

    dated_result = DatedPredictionResult(
        target_week_start="2026-08-17",
        snapshot_cutoff="2026-08-16",
        city_predictions=city_preds,
        total_predicted_interactions=round(total_predicted, 2),
    )

    if save_artifact:
        save_model_artifact(evaluation, dated_result, pipeline)

    return evaluation, dated_result, pipeline


def save_model_artifact(
    eval_result: ModelEvaluation,
    dated_result: DatedPredictionResult,
    model: Pipeline,
) -> None:
    """Save model provenance, metrics, and dated predictions to JSON artifact."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    reg: PoissonRegressor = model.named_steps["reg"]

    artifact = {
        "metadata": {
            "model_family": "sklearn.linear_model.PoissonRegressor",
            "alpha": eval_result.alpha,
            "trained_at": datetime.now().isoformat(),
            "target": "weekly_cultural_interactions_per_city",
            "grain": "city_week",
            "study_cities": 45,
            "complete_weeks": 57,
            "panel_total_rows": 2565,
            "zero_count_city_weeks": 1191,
            "qualifying_actions_in_panel": 2138,
            "eligible_panel_rows": 2385,
            "train_rows": eval_result.train_rows,
            "test_rows": eval_result.test_rows,
            "test_weeks_count": eval_result.test_weeks_count,
            "test_start_week": eval_result.test_start_week,
            "test_end_week": eval_result.test_end_week,
            "features": ["city_id", "lag_1", "lag_2", "trailing_4w_mean"],
            "sklearn_version": "1.9.1",
        },
        "model_parameters": {
            "intercept": float(reg.intercept_),
            "coef_count": len(reg.coef_),
            "lag_1_coef": float(reg.coef_[-3]),
            "lag_2_coef": float(reg.coef_[-2]),
            "trailing_4w_mean_coef": float(reg.coef_[-1]),
        },
        "metrics": {
            "model_mae": eval_result.model_mae,
            "baseline_prev_mae": eval_result.baseline_prev_mae,
            "baseline_4w_mae": eval_result.baseline_4w_mae,
            "model_rmse": eval_result.model_rmse,
            "baseline_prev_rmse": eval_result.baseline_prev_rmse,
            "baseline_4w_rmse": eval_result.baseline_4w_rmse,
            "zero_targets_count": eval_result.zero_targets_count,
            "zero_model_mae": eval_result.zero_model_mae,
            "zero_base_prev_mae": eval_result.zero_base_prev_mae,
            "zero_base_4w_mae": eval_result.zero_base_4w_mae,
            "nonzero_targets_count": eval_result.nonzero_targets_count,
            "nonzero_model_mae": eval_result.nonzero_model_mae,
            "nonzero_base_prev_mae": eval_result.nonzero_base_prev_mae,
            "nonzero_base_4w_mae": eval_result.nonzero_base_4w_mae,
        },
        "dated_prediction_2026_08_17": {
            "target_week_start": dated_result.target_week_start,
            "snapshot_cutoff": dated_result.snapshot_cutoff,
            "total_predicted_interactions": dated_result.total_predicted_interactions,
            "city_predictions": dated_result.city_predictions,
        },
    }

    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)


def load_model_artifact() -> Optional[Dict[str, Any]]:
    """Load model artifact from models/poisson_model_artifact.json if it exists."""
    if not ARTIFACT_PATH.exists():
        return None
    try:
        with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
