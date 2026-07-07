import struct

import pandas as pd

from src.const import CYCLE_COLUMN
from src_agent.config import LifePhaseBandsConfig
from src_agent.mcp_server.tools.database import fetch_dataframe
from src_agent.mcp_server.tools.sensor_names import resolve_feature_names

EARLY_PHASE = "early"
MID_PHASE = "mid"
LATE_PHASE = "late"
PHASE_ORDER = [EARLY_PHASE, MID_PHASE, LATE_PHASE]

TREND_DIRECTION_RELATIVE_CHANGE = 0.1


def assign_life_phase(predicted_life_ratio: float, bands: LifePhaseBandsConfig) -> str:
    """Map a predicted life_ratio to its life phase (early/mid/late)."""
    if predicted_life_ratio > bands.early_minimum_life_ratio:
        return EARLY_PHASE
    if predicted_life_ratio < bands.late_maximum_life_ratio:
        return LATE_PHASE
    return MID_PHASE


def fetch_shap_values_with_predictions(
    database_path: str,
    engine_id: int,
    feature_names: list[str] | None = None,
    cycle_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Fetch one engine's SHAP rows joined with the prediction of each cycle."""
    query = (
        "SELECT shap.cycle, shap.feature_name, shap.shap_value, "
        "readings.predicted_life_ratio "
        "FROM inference_shap_values AS shap "
        "JOIN inference_readings AS readings "
        "ON readings.engine_id = shap.engine_id AND readings.cycle = shap.cycle "
        "WHERE shap.engine_id = ?"
    )
    parameters: list = [engine_id]
    if feature_names:
        resolved_feature_names = resolve_feature_names(feature_names)
        placeholders = ", ".join("?" for _ in resolved_feature_names)
        query += f" AND shap.feature_name IN ({placeholders})"
        parameters.extend(resolved_feature_names)
    if cycle_range:
        query += " AND shap.cycle BETWEEN ? AND ?"
        parameters.extend(cycle_range)
    query += " ORDER BY shap.cycle"
    shap_values_df = fetch_dataframe(database_path, query, tuple(parameters))
    return _coerce_shap_value_column(shap_values_df)


def _coerce_shap_value_column(shap_values_df: pd.DataFrame) -> pd.DataFrame:
    """Decode SHAP values stored as BLOBs by older logging code.

    Numpy scalars used to reach sqlite3 uncast, which stores their raw bytes
    (4 bytes = float32, 8 = float64). Databases written since the fix hold
    REALs and pass straight through.
    """
    if shap_values_df["shap_value"].dtype == object:
        shap_values_df["shap_value"] = shap_values_df["shap_value"].map(
            _decode_stored_float
        )
    return shap_values_df


def _decode_stored_float(stored_value) -> float:
    if isinstance(stored_value, bytes):
        float_format = "<f" if len(stored_value) == 4 else "<d"
        return struct.unpack(float_format, stored_value)[0]
    return float(stored_value)


def compute_trend_summary(shap_values_df: pd.DataFrame) -> list[dict]:
    """Summarize each feature's SHAP contribution and its direction over time.

    Direction compares mean absolute SHAP between the first and second half
    of the cycle window: rising/falling when the relative change exceeds
    TREND_DIRECTION_RELATIVE_CHANGE, flat otherwise.
    """
    feature_summaries = []
    for feature_name, feature_df in shap_values_df.groupby("feature_name"):
        absolute_values = feature_df["shap_value"].abs().reset_index(drop=True)
        half_point = len(absolute_values) // 2
        first_half_mean = float(absolute_values.iloc[:half_point].mean())
        second_half_mean = float(absolute_values.iloc[half_point:].mean())
        feature_summaries.append(
            {
                "feature": feature_name,
                "mean_absolute_shap": float(absolute_values.mean()),
                "latest_shap_value": float(feature_df["shap_value"].iloc[-1]),
                "first_half_mean_absolute_shap": first_half_mean,
                "second_half_mean_absolute_shap": second_half_mean,
                "direction": _trend_direction(first_half_mean, second_half_mean),
            }
        )
    feature_summaries.sort(key=lambda entry: -entry["mean_absolute_shap"])
    return feature_summaries


def _trend_direction(first_half_mean: float, second_half_mean: float) -> str:
    """Classify the change between window halves as rising, falling, or flat."""
    if first_half_mean == 0:
        return "rising" if second_half_mean > 0 else "flat"
    relative_change = (second_half_mean - first_half_mean) / first_half_mean
    if relative_change > TREND_DIRECTION_RELATIVE_CHANGE:
        return "rising"
    if relative_change < -TREND_DIRECTION_RELATIVE_CHANGE:
        return "falling"
    return "flat"


def get_shap_trend(
    database_path: str,
    engine_id: int,
    feature_names: list[str] | None = None,
    cycle_range: tuple[int, int] | None = None,
) -> dict:
    """Report how each feature's SHAP contribution evolves for one engine."""
    shap_values_df = fetch_shap_values_with_predictions(
        database_path, engine_id, feature_names, cycle_range
    )
    if shap_values_df.empty:
        return {
            "engine_id": engine_id,
            "message": f"No logged SHAP values for engine {engine_id}",
            "features": [],
        }
    return {
        "engine_id": engine_id,
        "cycle_range": [
            int(shap_values_df[CYCLE_COLUMN].min()),
            int(shap_values_df[CYCLE_COLUMN].max()),
        ],
        "features": compute_trend_summary(shap_values_df),
    }


def compute_evidence_profile(
    shap_with_predictions_df: pd.DataFrame,
    bands: LifePhaseBandsConfig,
    top_feature_count: int,
) -> list[dict]:
    """Split SHAP rows into life phases and compute evidence shares per phase.

    For each phase, cycle_share is the fraction of total absolute SHAP carried
    by the literal `cycle` feature; sensor-derived features carry the rest.
    This quantifies the model's known behavior: cycle dominates at the start
    and end of life, sensor evidence peaks mid-life.
    """
    profile_df = shap_with_predictions_df.copy()
    profile_df["phase"] = profile_df["predicted_life_ratio"].map(
        lambda ratio: assign_life_phase(ratio, bands)
    )
    profile_df["absolute_shap"] = profile_df["shap_value"].abs()
    phase_profiles = []
    for phase_name in PHASE_ORDER:
        phase_df = profile_df[profile_df["phase"] == phase_name]
        if phase_df.empty:
            continue
        phase_profiles.append(
            _summarize_single_phase(phase_name, phase_df, top_feature_count)
        )
    return phase_profiles


def _summarize_single_phase(
    phase_name: str, phase_df: pd.DataFrame, top_feature_count: int
) -> dict:
    """Compute cycle-share vs top sensor-feature shares within one phase."""
    total_absolute_shap = float(phase_df["absolute_shap"].sum())
    shares_by_feature = (
        phase_df.groupby("feature_name")["absolute_shap"].sum() / total_absolute_shap
    ).sort_values(ascending=False)
    cycle_share = float(shares_by_feature.get(CYCLE_COLUMN, 0.0))
    sensor_shares = shares_by_feature.drop(CYCLE_COLUMN, errors="ignore")
    return {
        "phase": phase_name,
        "cycles_in_phase": int(phase_df[CYCLE_COLUMN].nunique()),
        "cycle_range": [
            int(phase_df[CYCLE_COLUMN].min()),
            int(phase_df[CYCLE_COLUMN].max()),
        ],
        "cycle_feature_share": cycle_share,
        "sensor_feature_share": float(sensor_shares.sum()),
        "top_sensor_features": [
            {"feature": feature_name, "share": float(share)}
            for feature_name, share in sensor_shares.head(top_feature_count).items()
        ],
    }


def get_shap_evidence_profile(
    database_path: str,
    engine_id: int,
    bands: LifePhaseBandsConfig,
    top_feature_count: int = 5,
) -> dict:
    """Report which evidence (cycle vs sensors) drives predictions per life phase.

    The headline diagnostic: tells the agent whether the engine currently sits
    in a cycle-dominated phase (prediction blind to sensor anomalies) or a
    sensor-driven phase (SHAP trends are credible degradation evidence).
    """
    shap_values_df = fetch_shap_values_with_predictions(database_path, engine_id)
    if shap_values_df.empty:
        return {
            "engine_id": engine_id,
            "message": f"No logged SHAP values for engine {engine_id}",
            "phases": [],
        }
    latest_cycle_df = shap_values_df[
        shap_values_df[CYCLE_COLUMN] == shap_values_df[CYCLE_COLUMN].max()
    ]
    latest_predicted_life_ratio = float(latest_cycle_df["predicted_life_ratio"].iloc[0])
    return {
        "engine_id": engine_id,
        "latest_cycle": int(shap_values_df[CYCLE_COLUMN].max()),
        "latest_predicted_life_ratio": latest_predicted_life_ratio,
        "current_phase": assign_life_phase(latest_predicted_life_ratio, bands),
        "phase_bands": bands.model_dump(),
        "phases": compute_evidence_profile(shap_values_df, bands, top_feature_count),
    }
