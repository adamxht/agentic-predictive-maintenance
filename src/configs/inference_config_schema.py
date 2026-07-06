import yaml
from pydantic import BaseModel, Field

from src.exception import CustomException

DEFAULT_INFERENCE_PIPELINE_STEPS = [
    "missing_value_handling",
    "scaling",
    "feature_engineering",
    "feature_selection",
]


class InferencePreprocessingConfig(BaseModel):
    """Real-time preprocessing settings -- must match what the model was trained on.

    No `drop_unused_columns`/target-computation steps: the model's selected
    features (from its bundled preprocessor/ folder) already say exactly
    which raw sensor columns matter, so InferencePipeline derives that
    restriction itself instead of needing a separate configured step.
    """

    pipeline_steps: list[str] = Field(
        default_factory=lambda: list(DEFAULT_INFERENCE_PIPELINE_STEPS)
    )
    rolling_window_size: int = 5
    lag_steps: list[int] = Field(default_factory=lambda: [1, 2])


class InferenceServingConfig(BaseModel):
    """Settings for the FastAPI inference service (app/api.py).

    Fully self-contained: unlike the test-set evaluation script, this does
    not read configs/data_transformation/default.yaml -- a deployed service
    shouldn't depend on a training-time config file.
    """

    model: str
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    database_path: str = "data/inference_log.db"
    life_ratio_threshold: float = 0.1
    preprocessing: InferencePreprocessingConfig = Field(
        default_factory=InferencePreprocessingConfig
    )


def load_inference_serving_config(config_path: str) -> InferenceServingConfig:
    """Load and validate the inference serving config from a YAML file."""
    try:
        with open(config_path) as config_file:
            raw_configuration = yaml.safe_load(config_file) or {}
        return InferenceServingConfig(**raw_configuration)
    except Exception as error:
        raise CustomException(str(error)) from error
