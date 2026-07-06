"""Stateless FastAPI service that scores real-time CMAPSS sensor readings.

Each request carries its own window of recent raw readings for one engine,
so the service holds no per-engine state between calls -- all history lives
on the caller's side. Every prediction is logged (raw sensor readings + SHAP
values) to a SQLite database for downstream monitoring/drift analysis.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse, ServingConfigResponse
from app.serving import (
    build_raw_window_dataframe,
    compute_required_window_length,
    compute_shap_values_dict,
)
from src.components import inference_store, model_loader
from src.configs.inference_config_schema import load_inference_serving_config
from src.exception import CustomException
from src.logger import logging
from src.pipeline.inference_pipeline import (
    CYCLE_COLUMN,
    ENGINE_ID_COLUMN,
    InferencePipeline,
)

SERVING_CONFIG_PATH = os.environ.get(
    "INFERENCE_CONFIG_PATH", "configs/deployment/default.yaml"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, its bundled preprocessor, and initialize the log database."""
    serving_config = load_inference_serving_config(SERVING_CONFIG_PATH)

    model = model_loader.load_model_for_evaluation(
        serving_config.model, serving_config.mlflow_tracking_uri
    )
    scaler, selected_features = model_loader.load_bundled_preprocessor(
        serving_config.model, serving_config.mlflow_tracking_uri
    )
    inference_pipeline = InferencePipeline(
        serving_config.preprocessing, scaler, selected_features
    )
    inference_store.initialize_database(
        serving_config.database_path, inference_pipeline.required_sensor_columns
    )

    app.state.serving_config = serving_config
    app.state.model = model
    app.state.sensor_columns = inference_pipeline.required_sensor_columns
    app.state.selected_features = selected_features
    app.state.inference_pipeline = inference_pipeline
    app.state.required_window_length = compute_required_window_length(
        serving_config.preprocessing.rolling_window_size,
        serving_config.preprocessing.lag_steps,
    )
    logging.info(f"Inference API ready, serving model: {serving_config.model}")
    yield


app = FastAPI(title="Predictive Maintenance Inference API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Report basic liveness."""
    return {"status": "ok"}


@app.get("/config", response_model=ServingConfigResponse)
def get_config() -> ServingConfigResponse:
    """Report the settings a client needs to build valid /predict requests."""
    return ServingConfigResponse(
        required_window_length=app.state.required_window_length,
        sensor_columns=app.state.sensor_columns,
        selected_features=app.state.selected_features,
        life_ratio_threshold=app.state.serving_config.life_ratio_threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Preprocess a reading window, score it, log it, and return the prediction."""
    try:
        raw_window = build_raw_window_dataframe(request)
        feature_row = app.state.inference_pipeline.run(raw_window)
        model_input = feature_row.drop(labels=[ENGINE_ID_COLUMN]).to_frame().T

        predicted_life_ratio = float(app.state.model.predict(model_input)[0])
        shap_values = compute_shap_values_dict(app.state.model, model_input)
        cycle = int(feature_row[CYCLE_COLUMN])

        raw_current_row = app.state.inference_pipeline.raw_dataframe.iloc[-1]
        sensor_readings = raw_current_row[app.state.sensor_columns].to_dict()
        inference_store.log_inference_reading(
            app.state.serving_config.database_path,
            request.engine_id,
            cycle,
            sensor_readings,
            predicted_life_ratio,
        )
        inference_store.log_shap_values(
            app.state.serving_config.database_path,
            request.engine_id,
            cycle,
            shap_values,
        )
        return PredictionResponse(
            engine_id=request.engine_id,
            cycle=cycle,
            predicted_life_ratio=predicted_life_ratio,
            shap_values=shap_values,
        )
    except CustomException as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.error(f"Prediction failed: {error}")
        raise HTTPException(status_code=500, detail=str(error)) from error
