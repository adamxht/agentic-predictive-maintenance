from typing import Literal

import yaml
from pydantic import BaseModel, Field

from src.exception import CustomException

DEFAULT_COLUMNS_TO_DROP = [
    "setting_1",
    "setting_2",
    "setting_3",
    "T2",
    "P2",
    "farB",
    "Nf_dmd",
    "PCNfR_dmd",
    "Nc",
    "P15",
    "P30",
    "epr",
]

DEFAULT_PIPELINE_STEPS = [
    "train_validation_split",
    "preprocessing",
    "missing_value_handling",
    "scaling",
    "feature_engineering",
    "feature_selection",
]

TARGET_COLUMN_BY_TYPE = {
    "rul": "RUL",
    "life_ratio": "life_ratio",
}


class DataPathsConfig(BaseModel):
    """File paths used by the data preparation pipeline."""

    raw_data_path: str = "data/raw/train_FD001.txt"
    processed_train_path: str = "data/processed/train.csv"
    processed_validation_path: str = "data/processed/val.csv"
    scaler_path: str = "data/processed/artifacts/scaler.pkl"
    selected_features_path: str = "data/processed/artifacts/selected_features.json"


class TrainValidationSplitConfig(BaseModel):
    """Settings for splitting engines into train and validation sets."""

    test_size: float = 0.2
    random_state: int = 42


class PreprocessingConfig(BaseModel):
    """Settings for cleaning raw sensor readings."""

    columns_to_drop: list[str] = Field(
        default_factory=lambda: list(DEFAULT_COLUMNS_TO_DROP)
    )


class MissingValueHandlingConfig(BaseModel):
    """Settings for imputing missing sensor readings."""

    method: Literal["interpolate_then_fill"] = "interpolate_then_fill"


class ScalingConfig(BaseModel):
    """Settings for scaling sensor features."""

    method: str = "standard"


class FeatureEngineeringConfig(BaseModel):
    """Settings for rolling window and lag feature generation."""

    rolling_window_size: int = 5
    lag_steps: list[int] = Field(default_factory=lambda: [1, 2])


class FeatureSelectionConfig(BaseModel):
    """Settings for correlation/variance-based feature selection."""

    top_k: int = 10


class PipelineStepsConfig(BaseModel):
    """Ordered list of data preparation steps to execute."""

    steps: list[str] = Field(default_factory=lambda: list(DEFAULT_PIPELINE_STEPS))


class TargetConfig(BaseModel):
    """Which label the pipeline prepares as the model's prediction target."""

    type: Literal["rul", "life_ratio"] = "life_ratio"

    @property
    def column_name(self) -> str:
        """Return the dataframe column name for the configured target type."""
        return TARGET_COLUMN_BY_TYPE[self.type]


class DataPreparationConfig(BaseModel):
    """Top-level configuration for the data preparation pipeline."""

    paths: DataPathsConfig = Field(default_factory=DataPathsConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    train_validation_split: TrainValidationSplitConfig = Field(
        default_factory=TrainValidationSplitConfig
    )
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    missing_value_handling: MissingValueHandlingConfig = Field(
        default_factory=MissingValueHandlingConfig
    )
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    pipeline: PipelineStepsConfig = Field(default_factory=PipelineStepsConfig)


def load_data_preparation_config(config_path: str) -> DataPreparationConfig:
    """Load and validate a data preparation config from a YAML file."""
    try:
        with open(config_path) as config_file:
            raw_configuration = yaml.safe_load(config_file) or {}
        return DataPreparationConfig(**raw_configuration)
    except Exception as error:
        raise CustomException(str(error)) from error
