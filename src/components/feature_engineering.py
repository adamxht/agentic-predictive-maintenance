import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging

ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"


def add_remaining_useful_life(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a RUL column computed as max_cycle - current_cycle for each engine."""
    try:
        dataframe = dataframe.copy()
        max_cycle_per_engine = dataframe.groupby(ENGINE_ID_COLUMN)[
            CYCLE_COLUMN
        ].transform("max")
        dataframe["RUL"] = max_cycle_per_engine - dataframe[CYCLE_COLUMN]
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def add_life_ratio(dataframe: pd.DataFrame, rul_column: str = "RUL") -> pd.DataFrame:
    """Add a life_ratio column (RUL / max_cycle), always in [0, 1], per engine."""
    try:
        dataframe = dataframe.copy()
        max_cycle_per_engine = dataframe.groupby(ENGINE_ID_COLUMN)[
            CYCLE_COLUMN
        ].transform("max")
        dataframe["life_ratio"] = dataframe[rul_column] / max_cycle_per_engine
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def drop_unused_columns(
    dataframe: pd.DataFrame, columns_to_drop: list[str]
) -> pd.DataFrame:
    """Drop configured columns from a dataframe, ignoring any that are absent."""
    try:
        return dataframe.drop(columns=columns_to_drop, errors="ignore")
    except Exception as error:
        raise CustomException(str(error)) from error


def handle_missing_sensor_values(
    dataframe: pd.DataFrame, sensor_columns: list[str]
) -> pd.DataFrame:
    """Fill missing sensor values via per-engine interpolation, then edge fill."""
    try:
        dataframe = dataframe.copy().sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN])
        missing_value_count = int(dataframe[sensor_columns].isna().sum().sum())
        if missing_value_count > 0:
            logging.warning(
                f"Filling {missing_value_count} missing sensor values via "
                "per-engine interpolation/edge-fill"
            )
        grouped_by_engine = dataframe.groupby(ENGINE_ID_COLUMN)[sensor_columns]
        dataframe[sensor_columns] = grouped_by_engine.transform(
            lambda series: series.interpolate(method="linear").bfill().ffill()
        )
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def fit_sensor_scaler(
    training_dataframe: pd.DataFrame, sensor_columns: list[str]
) -> StandardScaler:
    """Fit a StandardScaler on training sensor columns only."""
    try:
        scaler = StandardScaler()
        scaler.fit(training_dataframe[sensor_columns])
        logging.info(f"Fitted StandardScaler on {len(sensor_columns)} sensor columns")
        return scaler
    except Exception as error:
        raise CustomException(str(error)) from error


def apply_sensor_scaler(
    dataframe: pd.DataFrame, scaler: StandardScaler, sensor_columns: list[str]
) -> pd.DataFrame:
    """Apply an already-fitted scaler to the sensor columns of a dataframe."""
    try:
        dataframe = dataframe.copy()
        dataframe[sensor_columns] = scaler.transform(dataframe[sensor_columns])
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def add_rolling_window_features(
    dataframe: pd.DataFrame, sensor_columns: list[str], window_size: int
) -> pd.DataFrame:
    """Add a per-engine rolling mean feature for each sensor column."""
    try:
        dataframe = dataframe.copy().sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN])
        grouped_by_engine = dataframe.groupby(ENGINE_ID_COLUMN)
        for column in sensor_columns:
            dataframe[f"{column}_roll_mean"] = grouped_by_engine[column].transform(
                lambda series: series.rolling(window_size, min_periods=1).mean()
            )
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def add_lag_features(
    dataframe: pd.DataFrame, sensor_columns: list[str], lag_steps: list[int]
) -> pd.DataFrame:
    """Add per-engine lag features for each sensor column."""
    try:
        dataframe = dataframe.copy()
        grouped_by_engine = dataframe.groupby(ENGINE_ID_COLUMN)
        for column in sensor_columns:
            for lag in lag_steps:
                dataframe[f"{column}_lag{lag}"] = grouped_by_engine[column].shift(lag)
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def drop_rows_with_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop rows containing missing values, typically introduced by lag features."""
    try:
        result_dataframe = dataframe.dropna().reset_index(drop=True)
        dropped_row_count = len(dataframe) - len(result_dataframe)
        logging.info(f"Dropped {dropped_row_count} rows with missing values")
        return result_dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def select_top_features(
    training_dataframe: pd.DataFrame, target_column: str, top_k: int
) -> list[str]:
    """Select top-k feature columns ranked by |correlation| * variance vs. target."""
    try:
        numeric_dataframe = training_dataframe.select_dtypes(include=["number"])
        feature_columns = numeric_dataframe.columns.drop(
            [target_column], errors="ignore"
        )

        correlation = (
            numeric_dataframe[feature_columns]
            .corrwith(numeric_dataframe[target_column])
            .abs()
        )
        undefined_correlation_columns = correlation[correlation.isna()].index.tolist()
        if undefined_correlation_columns:
            logging.warning(
                "Zero-variance features produced undefined correlation, ranked "
                f"last: {undefined_correlation_columns}"
            )

        variance = numeric_dataframe[feature_columns].var()
        score = (correlation * variance).sort_values(ascending=False)

        if top_k == -1:
            selected_features = score.index.tolist()
        else:
            selected_features = score.head(top_k).index.tolist()

        logging.info(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    except Exception as error:
        raise CustomException(str(error)) from error


def apply_feature_selection(
    dataframe: pd.DataFrame, selected_features: list[str], target_column: str
) -> pd.DataFrame:
    """Keep only the selected features plus the target and engine/cycle identifiers."""
    try:
        identifier_columns = [CYCLE_COLUMN, ENGINE_ID_COLUMN]
        feature_and_identifier_columns = list(
            dict.fromkeys(selected_features + identifier_columns)
        )
        columns_to_keep = [target_column, *feature_and_identifier_columns]
        return dataframe[columns_to_keep].copy()
    except Exception as error:
        raise CustomException(str(error)) from error
