import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.components import feature_engineering
from src.configs.inference_config_schema import InferencePreprocessingConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import get_sensor_columns

ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"
IDENTIFIER_COLUMNS = (ENGINE_ID_COLUMN, CYCLE_COLUMN)


def derive_required_sensor_columns(scaler: StandardScaler) -> list[str]:
    """Return the raw sensor columns the model's bundled scaler was fit on.

    This is the single source of truth for which raw sensor columns a
    reading window must contain: it's already known from the bundled
    preprocessor, so no separate drop-columns config is needed to restrict
    an incoming window before scaling/feature engineering run.
    """
    return list(scaler.feature_names_in_)


class InferencePipeline:
    """Turns a raw sensor reading window into one model-ready feature row.

    Stateless by design: every call takes the full window of recent raw
    readings for one engine (oldest to newest) and recomputes rolling/lag
    features from scratch, holding no per-engine history between calls. The
    caller (e.g. the serving API) owns buffering that window; the scaler and
    selected-feature list are supplied already loaded, from the serving
    model's own bundled preprocessor/ folder. There's no separate
    drop-columns step: the scaler's own fit-time columns say exactly which
    raw sensors are needed, and the final feature_selection step narrows the
    post-engineering columns down to the model's exact selected features.
    """

    def __init__(
        self,
        configuration: InferencePreprocessingConfig,
        scaler: StandardScaler,
        selected_features: list[str],
    ) -> None:
        self.configuration = configuration
        self.scaler = scaler
        self.selected_features = selected_features
        self.required_sensor_columns = derive_required_sensor_columns(scaler)
        self.dataframe: pd.DataFrame | None = None
        self.raw_dataframe: pd.DataFrame | None = None
        self._step_registry = {
            "missing_value_handling": self._run_missing_value_handling,
            "scaling": self._run_scaling,
            "feature_engineering": self._run_feature_engineering,
            "feature_selection": self._run_feature_selection,
        }

    def run(self, raw_window: pd.DataFrame) -> pd.Series:
        """Preprocess a raw reading window and return the latest cycle's feature row.

        raw_window must contain at least engine_id, cycle, and every column
        required_sensor_columns lists, with one row per recent cycle for a
        single engine.
        """
        try:
            sorted_window = raw_window.sort_values(
                [ENGINE_ID_COLUMN, CYCLE_COLUMN]
            ).reset_index(drop=True)
            self.dataframe = sorted_window[
                [*IDENTIFIER_COLUMNS, *self.required_sensor_columns]
            ]
            self.raw_dataframe = self.dataframe.copy()

            for step_name in self.configuration.pipeline_steps:
                if step_name not in self._step_registry:
                    raise ValueError(f"Unknown inference pipeline step: {step_name}")
                self._step_registry[step_name]()

            if self.dataframe.empty:
                raise ValueError(
                    "No cycle in the window has enough history for rolling/lag "
                    "features -- provide a longer window."
                )
            return self.dataframe.iloc[-1]
        except Exception as error:
            logging.error(f"Inference preprocessing failed: {error}")
            raise CustomException(str(error)) from error

    def _run_missing_value_handling(self) -> None:
        """Fill missing sensor values the same way as the training pipeline."""
        sensor_columns = get_sensor_columns(self.dataframe)
        self.dataframe = feature_engineering.handle_missing_sensor_values(
            self.dataframe, sensor_columns
        )

    def _run_scaling(self) -> None:
        """Apply the model's bundled scaler; never fit on live data."""
        sensor_columns = get_sensor_columns(self.dataframe)
        self.dataframe = feature_engineering.apply_sensor_scaler(
            self.dataframe, self.scaler, sensor_columns
        )

    def _run_feature_engineering(self) -> None:
        """Add rolling window and lag features, then drop rows with missing values."""
        sensor_columns = get_sensor_columns(self.dataframe)
        window_size = self.configuration.rolling_window_size
        lag_steps = self.configuration.lag_steps

        self.dataframe = feature_engineering.add_rolling_window_features(
            self.dataframe, sensor_columns, window_size
        )
        self.dataframe = feature_engineering.add_lag_features(
            self.dataframe, sensor_columns, lag_steps
        )
        self.dataframe = feature_engineering.drop_rows_with_missing_values(
            self.dataframe
        )

    def _run_feature_selection(self) -> None:
        """Apply the model's bundled feature list; never reselect on live data."""
        self.dataframe = feature_engineering.apply_feature_selection(
            self.dataframe, self.selected_features
        )
