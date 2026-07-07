import pandas as pd

from src.const import CYCLE_COLUMN
from src_agent.mcp_server.tools.database import fetch_dataframe


def fetch_prediction_series(
    database_path: str,
    engine_id: int,
    cycle_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Fetch one engine's predicted life_ratio per cycle, oldest first."""
    query = (
        "SELECT cycle, predicted_life_ratio FROM inference_readings WHERE engine_id = ?"
    )
    parameters: list = [engine_id]
    if cycle_range:
        query += " AND cycle BETWEEN ? AND ?"
        parameters.extend(cycle_range)
    query += " ORDER BY cycle"
    return fetch_dataframe(database_path, query, tuple(parameters))


def summarize_prediction_trend(
    prediction_series_df: pd.DataFrame, recent_window_size: int
) -> dict:
    """Summarize a prediction series: latest value, recent change, and extremes."""
    predicted_values = prediction_series_df["predicted_life_ratio"]
    recent_values = predicted_values.tail(recent_window_size)
    return {
        "latest_cycle": int(prediction_series_df[CYCLE_COLUMN].iloc[-1]),
        "latest_predicted_life_ratio": float(predicted_values.iloc[-1]),
        "recent_change": float(recent_values.iloc[-1] - recent_values.iloc[0]),
        "recent_window_size": len(recent_values),
        "minimum_predicted_life_ratio": float(predicted_values.min()),
        "maximum_predicted_life_ratio": float(predicted_values.max()),
    }


def get_prediction_trend(
    database_path: str,
    engine_id: int,
    cycle_range: tuple[int, int] | None = None,
    recent_window_size: int = 10,
) -> dict:
    """Report an engine's predicted life_ratio series plus a trend summary."""
    prediction_series_df = fetch_prediction_series(
        database_path, engine_id, cycle_range
    )
    if prediction_series_df.empty:
        return {
            "engine_id": engine_id,
            "message": f"No logged predictions for engine {engine_id}",
            "series": [],
        }
    return {
        "engine_id": engine_id,
        "summary": summarize_prediction_trend(prediction_series_df, recent_window_size),
        "series": [
            {
                "cycle": int(row.cycle),
                "predicted_life_ratio": float(row.predicted_life_ratio),
            }
            for row in prediction_series_df.itertuples()
        ],
    }
