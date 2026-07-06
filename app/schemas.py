"""Pydantic request/response schemas for the inference API (app/api.py)."""

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One raw sensor reading for a single engine cycle."""

    cycle: int
    values: dict[str, float] = Field(
        ..., description="Raw sensor column name -> reading value"
    )


class PredictionRequest(BaseModel):
    """A window of raw sensor readings for one engine, ordered oldest to newest."""

    engine_id: int
    readings: list[SensorReading]


class PredictionResponse(BaseModel):
    """The model's prediction and SHAP explanation for the latest cycle."""

    engine_id: int
    cycle: int
    predicted_life_ratio: float
    shap_values: dict[str, float]


class ServingConfigResponse(BaseModel):
    """Settings a client needs to build valid /predict requests."""

    required_window_length: int
    sensor_columns: list[str]
    selected_features: list[str]
    life_ratio_threshold: float
