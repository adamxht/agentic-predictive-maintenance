"""Request-building and response-shaping helpers for the inference API.

Kept separate from app/api.py so that module only holds route handlers and
the app's lifespan.
"""

import pandas as pd

from app.schemas import PredictionRequest
from src.components import explain
from src.pipeline.inference_pipeline import CYCLE_COLUMN, ENGINE_ID_COLUMN


def compute_required_window_length(
    rolling_window_size: int, lag_steps: list[int]
) -> int:
    """Return the minimum reading-window length with no NaN in the latest cycle."""
    max_lag = max(lag_steps, default=0)
    return max(rolling_window_size, max_lag + 1)


def build_raw_window_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """Turn a prediction request's readings into a raw sensor dataframe."""
    rows = [
        {
            ENGINE_ID_COLUMN: request.engine_id,
            CYCLE_COLUMN: reading.cycle,
            **reading.values,
        }
        for reading in request.readings
    ]
    return pd.DataFrame(rows)


def compute_shap_values_dict(
    model: object, model_input: pd.DataFrame
) -> dict[str, float]:
    """Compute per-feature SHAP values for a single-row model input."""
    shap_explanation = explain.compute_shap_values(model, model_input)
    return dict(zip(model_input.columns, shap_explanation.values[0], strict=True))
