import json
import os

import pandas as pd

from src.exception import CustomException
from src_agent.mcp_server.tools.database import fetch_dataframe, fetch_table_columns
from src_agent.mcp_server.tools.sensor_names import resolve_feature_names

INFERENCE_READINGS_TABLE = "inference_readings"

MISSING_STATISTICS_HINT = (
    "Training statistics file not found at '{path}'. Generate it with "
    "`python scripts/run_training_stats.py` before starting the MCP server."
)


def load_training_statistics(statistics_path: str) -> dict[str, dict[str, float]]:
    """Load the precomputed per-sensor training statistics (mean/std/min/max)."""
    if not os.path.exists(statistics_path):
        raise CustomException(MISSING_STATISTICS_HINT.format(path=statistics_path))
    try:
        with open(statistics_path) as statistics_file:
            statistics_document = json.load(statistics_file)
        return statistics_document["statistics"]
    except Exception as error:
        raise CustomException(str(error)) from error


def fetch_recent_readings(
    database_path: str, engine_id: int, sensor_names: list[str], window_size: int
) -> pd.DataFrame:
    """Fetch the latest window_size raw readings for one engine, oldest first."""
    quoted_columns = ", ".join(f'"{name}"' for name in sensor_names)
    query = (
        f"SELECT cycle, {quoted_columns} FROM inference_readings "
        "WHERE engine_id = ? ORDER BY cycle DESC LIMIT ?"
    )
    recent_readings_df = fetch_dataframe(database_path, query, (engine_id, window_size))
    return recent_readings_df.sort_values("cycle").reset_index(drop=True)


def compute_sensor_drift(
    sensor_readings_df: pd.DataFrame,
    training_statistics: dict[str, dict[str, float]],
    sensor_names: list[str],
    alert_threshold: float,
) -> list[dict]:
    """Compute z-scores of recent readings against the training distribution.

    Returns one summary per sensor: training mean/std, the window mean, the
    latest value, both expressed as z-scores, and an out_of_distribution flag
    raised when either z-score magnitude reaches alert_threshold.
    """
    drift_summaries = []
    for sensor_name in sensor_names:
        sensor_statistics = training_statistics[sensor_name]
        sensor_values = sensor_readings_df[sensor_name].astype(float)
        drift_summaries.append(
            _summarize_single_sensor(
                sensor_name, sensor_values, sensor_statistics, alert_threshold
            )
        )
    return drift_summaries


def _summarize_single_sensor(
    sensor_name: str,
    sensor_values: pd.Series,
    sensor_statistics: dict[str, float],
    alert_threshold: float,
) -> dict:
    """Build the drift summary for one sensor's window of values."""
    training_mean = sensor_statistics["mean"]
    training_std = sensor_statistics["std"]
    window_mean = float(sensor_values.mean())
    latest_value = float(sensor_values.iloc[-1])
    latest_z_score = _z_score(latest_value, training_mean, training_std)
    window_mean_z_score = _z_score(window_mean, training_mean, training_std)
    out_of_distribution = any(
        z_score is not None and abs(z_score) >= alert_threshold
        for z_score in (latest_z_score, window_mean_z_score)
    )
    return {
        "sensor": sensor_name,
        "training_mean": training_mean,
        "training_std": training_std,
        "window_mean": window_mean,
        "latest_value": latest_value,
        "latest_z_score": latest_z_score,
        "window_mean_z_score": window_mean_z_score,
        "out_of_distribution": out_of_distribution,
    }


def _z_score(value: float, mean: float, std: float) -> float | None:
    """Standard z-score; None when the training std is zero (constant sensor)."""
    if std == 0:
        return None
    return (value - mean) / std


def compare_to_training_distribution(
    database_path: str,
    training_statistics: dict[str, dict[str, float]],
    engine_id: int,
    sensor_names: list[str] | None,
    window_size: int,
    alert_threshold: float,
) -> dict:
    """Compare an engine's recent raw readings against the training distribution.

    When sensor_names is None every sensor with both training statistics and
    a logged column is checked (the statistics file covers the full raw
    sensor set, which may be a superset of what a given deployment logs).
    Sensors that fail either check are reported separately instead of
    failing the whole call.
    """
    logged_columns = fetch_table_columns(database_path, INFERENCE_READINGS_TABLE)
    requested_sensors = (
        resolve_feature_names(sensor_names)
        if sensor_names
        else sorted(set(training_statistics.keys()) & logged_columns)
    )
    known_sensors = [
        name
        for name in requested_sensors
        if name in training_statistics and name in logged_columns
    ]
    unknown_sensors = [name for name in requested_sensors if name not in known_sensors]
    if not known_sensors:
        raise CustomException(
            f"None of the requested sensors {requested_sensors} have training "
            "statistics"
        )
    recent_readings_df = fetch_recent_readings(
        database_path, engine_id, known_sensors, window_size
    )
    if recent_readings_df.empty:
        return {
            "engine_id": engine_id,
            "message": f"No logged readings for engine {engine_id}",
            "sensors": [],
            "unknown_sensors": unknown_sensors,
        }
    return {
        "engine_id": engine_id,
        "window_size": len(recent_readings_df),
        "cycle_range": [
            int(recent_readings_df["cycle"].min()),
            int(recent_readings_df["cycle"].max()),
        ],
        "alert_threshold": alert_threshold,
        "sensors": compute_sensor_drift(
            recent_readings_df, training_statistics, known_sensors, alert_threshold
        ),
        "unknown_sensors": unknown_sensors,
    }
