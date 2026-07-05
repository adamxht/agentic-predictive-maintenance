from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.config_schema import TargetConfig
from src.exception import CustomException

DEFAULT_RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 600},
    "max_depth": {"type": "int", "low": 5, "high": 100},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    "max_features": {
        "type": "categorical",
        "choices": ["sqrt", "log2", 0.5, 0.8, None],
    },
    "bootstrap": {"type": "categorical", "choices": [True, False]},
}

DEFAULT_XGBOOST_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 0.0, "high": 5.0},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 5.0},
    "reg_lambda": {"type": "float", "low": 0.0, "high": 5.0},
}


class HyperparameterSpec(BaseModel):
    """A single hyperparameter's Optuna search space."""

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list | None = None


class ModelConfig(BaseModel):
    """Settings for tuning and training one model."""

    name: Literal["random_forest", "xgboost"]
    n_trials: int = 100
    fixed_params: dict = Field(default_factory=dict)
    search_space: dict[str, HyperparameterSpec] = Field(default_factory=dict)
    registered_model_name: str | None = None


class ModelTrainingDataConfig(BaseModel):
    """Paths to the processed data consumed by the training pipeline."""

    processed_train_path: str = "data/processed/train.csv"
    processed_validation_path: str = "data/processed/val.csv"


class BinaryClassificationConfig(BaseModel):
    """Settings for deriving a near-failure binary label from continuous predictions."""

    threshold: float = 0.1
    pred_offset: float = 0.0


class ExplainabilityConfig(BaseModel):
    """Settings for SHAP-based model explanations."""

    sample_size: int = 10
    random_state: int = 42


class PlotsConfig(BaseModel):
    """Settings for saving evaluation/explainability plots."""

    enabled: bool = True
    output_dir: str = "training_logs"


class MLflowConfig(BaseModel):
    """Settings for MLflow experiment tracking."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "cmapss_life_ratio"


class ModelTrainingConfig(BaseModel):
    """Top-level configuration for the model training pipeline."""

    run_name: str = "baseline"
    data: ModelTrainingDataConfig = Field(default_factory=ModelTrainingDataConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(
                name="random_forest",
                n_trials=100,
                fixed_params={"random_state": 42, "n_jobs": -1},
                search_space=DEFAULT_RANDOM_FOREST_SEARCH_SPACE,
            ),
            ModelConfig(
                name="xgboost",
                n_trials=100,
                fixed_params={
                    "random_state": 42,
                    "n_jobs": -1,
                    "objective": "reg:squarederror",
                },
                search_space=DEFAULT_XGBOOST_SEARCH_SPACE,
            ),
        ]
    )
    binary_classification: BinaryClassificationConfig = Field(
        default_factory=BinaryClassificationConfig
    )
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @model_validator(mode="after")
    def _apply_default_registered_model_names(self) -> "ModelTrainingConfig":
        """Default each model's registered_model_name to <run_name>_<model_name>."""
        for model_config in self.models:
            if model_config.registered_model_name is None:
                model_config.registered_model_name = (
                    f"{self.run_name}_{model_config.name}"
                )
        return self


def load_model_training_config(config_path: str) -> ModelTrainingConfig:
    """Load and validate a model training config from a YAML file."""
    try:
        with open(config_path) as config_file:
            raw_configuration = yaml.safe_load(config_file) or {}
        return ModelTrainingConfig(**raw_configuration)
    except Exception as error:
        raise CustomException(str(error)) from error
