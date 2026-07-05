import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"


def load_raw_sensor_readings(file_path: str, sensor_columns: list[str]) -> pd.DataFrame:
    """Load whitespace-delimited raw CMAPSS sensor readings into a dataframe."""
    try:
        columns = [ENGINE_ID_COLUMN, CYCLE_COLUMN, *sensor_columns]
        dataframe = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            usecols=range(len(columns)),
            names=columns,
            engine="python",
        )
        logging.info(
            f"Loaded raw sensor readings with shape {dataframe.shape} from {file_path}"
        )
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def split_train_validation_by_engine(
    dataframe: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation sets by engine_id to avoid leakage."""
    try:
        engine_ids = dataframe[ENGINE_ID_COLUMN].unique()
        train_engine_ids, validation_engine_ids = train_test_split(
            engine_ids, test_size=test_size, random_state=random_state
        )

        train_dataframe = _select_and_sort_engines(dataframe, train_engine_ids)
        validation_dataframe = _select_and_sort_engines(
            dataframe, validation_engine_ids
        )

        logging.info(
            f"Split into {train_dataframe[ENGINE_ID_COLUMN].nunique()} train engines "
            f"and {validation_dataframe[ENGINE_ID_COLUMN].nunique()} validation engines"
        )
        return train_dataframe, validation_dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def _select_and_sort_engines(dataframe: pd.DataFrame, engine_ids) -> pd.DataFrame:
    """Select rows for the given engine ids and sort by engine_id then cycle."""
    selected_rows = dataframe[dataframe[ENGINE_ID_COLUMN].isin(engine_ids)].copy()
    return selected_rows.sort_values([ENGINE_ID_COLUMN, CYCLE_COLUMN]).reset_index(
        drop=True
    )
