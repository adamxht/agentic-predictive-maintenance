from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.components import data_ingestion, feature_engineering, test_set_ingestion
from src.configs.data_pipeline_config_schema import DataPreparationConfig
from src.const import SENSOR_NAMES
from src.exception import CustomException
from src.logger import logging
from src.utils import (
    get_sensor_columns,
    load_json,
    load_object,
    save_dataframe,
    save_json,
    save_object,
)


@dataclass
class DataPreparationArtifacts:
    """Outputs produced by running the data preparation pipeline."""

    train_dataframe: pd.DataFrame | None = None
    validation_dataframe: pd.DataFrame | None = None
    scaler: StandardScaler | None = None
    selected_features: list[str] | None = None


@dataclass
class TestSetPreparationArtifacts(DataPreparationArtifacts):
    """Outputs produced by running the test-set preparation pipeline."""

    test_dataframe: pd.DataFrame | None = None


class DataPreparationPipeline:
    """Chains configurable data preparation steps for the CMAPSS RUL dataset."""

    def __init__(self, configuration: DataPreparationConfig) -> None:
        self.configuration = configuration
        self.train_dataframe: pd.DataFrame | None = None
        self.validation_dataframe: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None
        self.selected_features: list[str] | None = None
        self._step_registry = {
            "train_validation_split": self._run_train_validation_split,
            "preprocessing": self._run_preprocessing,
            "missing_value_handling": self._run_missing_value_handling,
            "scaling": self._run_scaling,
            "feature_engineering": self._run_feature_engineering,
            "feature_selection": self._run_feature_selection,
        }

    def run(self) -> DataPreparationArtifacts:
        """Execute the configured pipeline steps in order and persist the outputs."""
        try:
            for step_name in self.configuration.pipeline.steps:
                if step_name not in self._step_registry:
                    raise ValueError(f"Unknown pipeline step: {step_name}")
                logging.info(f"Running data preparation step: {step_name}")
                self._step_registry[step_name]()
            self._save_outputs()
            return DataPreparationArtifacts(
                train_dataframe=self.train_dataframe,
                validation_dataframe=self.validation_dataframe,
                scaler=self.scaler,
                selected_features=self.selected_features,
            )
        except Exception as error:
            logging.error(f"Data preparation pipeline failed: {error}")
            raise CustomException(str(error)) from error

    def _run_train_validation_split(self) -> None:
        """Load the raw sensor readings and split them into train/validation splits."""
        raw_dataframe = data_ingestion.load_raw_sensor_readings(
            self.configuration.paths.raw_data_path, SENSOR_NAMES
        )
        split_config = self.configuration.train_validation_split
        self.train_dataframe, self.validation_dataframe = (
            data_ingestion.split_train_validation_by_engine(
                raw_dataframe, split_config.test_size, split_config.random_state
            )
        )

    def _run_preprocessing(self) -> None:
        """Add the configured target column and drop configured columns from both."""
        columns_to_drop = self.configuration.preprocessing.columns_to_drop
        target_type = self.configuration.target.type
        for split_name in ("train_dataframe", "validation_dataframe"):
            dataframe = getattr(self, split_name)
            dataframe = feature_engineering.add_remaining_useful_life(dataframe)
            if target_type == "life_ratio":
                dataframe = feature_engineering.add_life_ratio(dataframe)
                dataframe = feature_engineering.drop_unused_columns(dataframe, ["RUL"])
            dataframe = feature_engineering.drop_unused_columns(
                dataframe, columns_to_drop
            )
            setattr(self, split_name, dataframe)

    def _run_missing_value_handling(self) -> None:
        """Fill missing sensor values independently within each split."""
        if self.configuration.missing_value_handling.method != "interpolate_then_fill":
            raise ValueError(
                "Unsupported missing value handling method: "
                f"{self.configuration.missing_value_handling.method}"
            )

        for split_name in ("train_dataframe", "validation_dataframe"):
            dataframe = getattr(self, split_name)
            sensor_columns = get_sensor_columns(dataframe)
            dataframe = feature_engineering.handle_missing_sensor_values(
                dataframe, sensor_columns
            )
            setattr(self, split_name, dataframe)

    def _run_scaling(self) -> None:
        """Fit a scaler on the training split and apply it to both splits."""
        if self.configuration.scaling.method != "standard":
            raise ValueError(
                f"Unsupported scaling method: {self.configuration.scaling.method}"
            )

        sensor_columns = get_sensor_columns(self.train_dataframe)
        self.scaler = feature_engineering.fit_sensor_scaler(
            self.train_dataframe, sensor_columns
        )
        self.train_dataframe = feature_engineering.apply_sensor_scaler(
            self.train_dataframe, self.scaler, sensor_columns
        )
        self.validation_dataframe = feature_engineering.apply_sensor_scaler(
            self.validation_dataframe, self.scaler, sensor_columns
        )

    def _run_feature_engineering(self) -> None:
        """Add rolling window and lag features, then drop rows with missing values."""
        sensor_columns = get_sensor_columns(self.train_dataframe)
        window_size = self.configuration.feature_engineering.rolling_window_size
        lag_steps = self.configuration.feature_engineering.lag_steps

        for split_name in ("train_dataframe", "validation_dataframe"):
            dataframe = getattr(self, split_name)
            dataframe = feature_engineering.add_rolling_window_features(
                dataframe, sensor_columns, window_size
            )
            dataframe = feature_engineering.add_lag_features(
                dataframe, sensor_columns, lag_steps
            )
            dataframe = feature_engineering.drop_rows_with_missing_values(dataframe)
            setattr(self, split_name, dataframe)

    def _run_feature_selection(self) -> None:
        """Select the top-k training features and apply them to both splits."""
        target_column = self.configuration.target.column_name
        top_k = self.configuration.feature_selection.top_k

        self.selected_features = feature_engineering.select_top_features(
            self.train_dataframe, target_column, top_k
        )
        self.train_dataframe = feature_engineering.apply_feature_selection(
            self.train_dataframe, self.selected_features, target_column
        )
        self.validation_dataframe = feature_engineering.apply_feature_selection(
            self.validation_dataframe, self.selected_features, target_column
        )

    def _save_outputs(self) -> None:
        """Persist the processed train/validation splits and any fitted artifacts."""
        paths = self.configuration.paths
        save_dataframe(self.train_dataframe, paths.processed_train_path)
        save_dataframe(self.validation_dataframe, paths.processed_validation_path)
        if self.scaler is not None:
            save_object(paths.scaler_path, self.scaler)
        if self.selected_features is not None:
            save_json(self.selected_features, paths.selected_features_path)


class TestSetPreparationPipeline:
    """Prepares the held-out, censored CMAPSS test set for evaluation.

    Reuses the scaler and selected-feature list already fitted/selected by
    DataPreparationPipeline on the training split -- must be run after it.
    """

    def __init__(self, configuration: DataPreparationConfig) -> None:
        self.configuration = configuration
        self.test_dataframe: pd.DataFrame | None = None
        self._step_registry = {
            "test_set_ingestion": self._run_test_set_ingestion,
            "missing_value_handling": self._run_missing_value_handling,
            "scaling": self._run_scaling,
            "feature_engineering": self._run_feature_engineering,
            "feature_selection": self._run_feature_selection,
        }

    def run(self) -> TestSetPreparationArtifacts:
        """Run the configured test-set pipeline steps in order and save the output."""
        try:
            for step_name in self.configuration.test_set.pipeline.steps:
                if step_name not in self._step_registry:
                    raise ValueError(f"Unknown test-set pipeline step: {step_name}")
                logging.info(f"Running test-set preparation step: {step_name}")
                self._step_registry[step_name]()
            save_dataframe(
                self.test_dataframe, self.configuration.test_set.processed_test_path
            )
            return TestSetPreparationArtifacts(test_dataframe=self.test_dataframe)
        except Exception as error:
            logging.error(f"Test-set preparation pipeline failed: {error}")
            raise CustomException(str(error)) from error

    def _run_test_set_ingestion(self) -> None:
        """Load the raw censored test set and reconstruct its target column."""
        test_set_config = self.configuration.test_set
        raw_dataframe = data_ingestion.load_raw_sensor_readings(
            test_set_config.raw_data_path, SENSOR_NAMES
        )
        terminal_rul_values = test_set_ingestion.load_raw_terminal_rul(
            test_set_config.raw_rul_path
        )
        dataframe = test_set_ingestion.add_censored_remaining_useful_life(
            raw_dataframe, terminal_rul_values
        )
        if self.configuration.target.type == "life_ratio":
            dataframe = test_set_ingestion.add_censored_life_ratio(dataframe)
            dataframe = feature_engineering.drop_unused_columns(dataframe, ["RUL"])

        columns_to_drop = [
            *self.configuration.preprocessing.columns_to_drop,
            test_set_ingestion.TERMINAL_RUL_COLUMN,
        ]
        self.test_dataframe = feature_engineering.drop_unused_columns(
            dataframe, columns_to_drop
        )

    def _run_missing_value_handling(self) -> None:
        """Fill missing sensor values the same way as the training pipeline."""
        if self.configuration.missing_value_handling.method != "interpolate_then_fill":
            raise ValueError(
                "Unsupported missing value handling method: "
                f"{self.configuration.missing_value_handling.method}"
            )
        sensor_columns = get_sensor_columns(self.test_dataframe)
        self.test_dataframe = feature_engineering.handle_missing_sensor_values(
            self.test_dataframe, sensor_columns
        )

    def _run_scaling(self) -> None:
        """Apply the scaler fitted on the training split; never refit on test data."""
        if self.configuration.scaling.method != "standard":
            raise ValueError(
                f"Unsupported scaling method: {self.configuration.scaling.method}"
            )
        scaler = load_object(self.configuration.paths.scaler_path)
        sensor_columns = get_sensor_columns(self.test_dataframe)
        self.test_dataframe = feature_engineering.apply_sensor_scaler(
            self.test_dataframe, scaler, sensor_columns
        )

    def _run_feature_engineering(self) -> None:
        """Add rolling window and lag features, then drop rows with missing values."""
        sensor_columns = get_sensor_columns(self.test_dataframe)
        window_size = self.configuration.feature_engineering.rolling_window_size
        lag_steps = self.configuration.feature_engineering.lag_steps

        self.test_dataframe = feature_engineering.add_rolling_window_features(
            self.test_dataframe, sensor_columns, window_size
        )
        self.test_dataframe = feature_engineering.add_lag_features(
            self.test_dataframe, sensor_columns, lag_steps
        )
        self.test_dataframe = feature_engineering.drop_rows_with_missing_values(
            self.test_dataframe
        )

    def _run_feature_selection(self) -> None:
        """Apply the feature list selected on the training split; never reselect."""
        selected_features = load_json(self.configuration.paths.selected_features_path)
        target_column = self.configuration.target.column_name
        self.test_dataframe = feature_engineering.apply_feature_selection(
            self.test_dataframe, selected_features, target_column
        )
