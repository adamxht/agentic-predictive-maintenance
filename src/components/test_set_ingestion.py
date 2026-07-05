import pandas as pd

from src.exception import CustomException
from src.logger import logging

ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"
TERMINAL_RUL_COLUMN = "_terminal_rul"


def load_raw_terminal_rul(file_path: str) -> pd.Series:
    """Load the terminal RUL answer key (one value per engine) for the test set."""
    try:
        terminal_rul = pd.read_csv(file_path, sep=r"\s+", header=None).iloc[:, 0]
        logging.info(f"Loaded {len(terminal_rul)} terminal RUL values from {file_path}")
        return terminal_rul
    except Exception as error:
        raise CustomException(str(error)) from error


def add_censored_remaining_useful_life(
    dataframe: pd.DataFrame, terminal_rul_values: pd.Series
) -> pd.DataFrame:
    """Reconstruct RUL for a censored test set using the terminal RUL answer key.

    The test set stops before failure, so RUL can't be derived from max_cycle
    alone (unlike add_remaining_useful_life). Instead, RUL at each row is the
    provided terminal RUL plus however many cycles remained after that row's
    cycle. Keeps the per-engine terminal RUL as a helper column for
    add_censored_life_ratio.
    """
    try:
        dataframe = dataframe.copy()
        engine_ids = dataframe[ENGINE_ID_COLUMN].unique()
        if len(terminal_rul_values) != len(engine_ids):
            raise ValueError(
                f"Expected {len(engine_ids)} terminal RUL values, got "
                f"{len(terminal_rul_values)}"
            )

        terminal_rul_by_engine = dict(
            zip(engine_ids, terminal_rul_values.tolist(), strict=True)
        )
        dataframe[TERMINAL_RUL_COLUMN] = dataframe[ENGINE_ID_COLUMN].map(
            terminal_rul_by_engine
        )
        last_cycle = dataframe.groupby(ENGINE_ID_COLUMN)[CYCLE_COLUMN].transform("max")
        dataframe["RUL"] = dataframe[TERMINAL_RUL_COLUMN] + (
            last_cycle - dataframe[CYCLE_COLUMN]
        )
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error


def add_censored_life_ratio(
    dataframe: pd.DataFrame, rul_column: str = "RUL"
) -> pd.DataFrame:
    """Add life_ratio for a censored test set: RUL / (last_cycle + terminal RUL).

    Unlike add_life_ratio (which divides by max_cycle), the denominator here is
    the reconstructed total life, since max_cycle is just the last *observed*
    cycle, not the actual failure cycle. Requires
    add_censored_remaining_useful_life to have been applied first.
    """
    try:
        dataframe = dataframe.copy()
        last_cycle = dataframe.groupby(ENGINE_ID_COLUMN)[CYCLE_COLUMN].transform("max")
        total_life = last_cycle + dataframe[TERMINAL_RUL_COLUMN]
        dataframe["life_ratio"] = dataframe[rul_column] / total_life
        return dataframe
    except Exception as error:
        raise CustomException(str(error)) from error
