import os
import sqlite3
from datetime import UTC, datetime

from src.exception import CustomException
from src.logger import logging

INFERENCE_READINGS_TABLE = "inference_readings"
SHAP_VALUES_TABLE = "inference_shap_values"


def initialize_database(database_path: str, sensor_columns: list[str]) -> None:
    """Create the inference logging tables if they don't already exist.

    inference_readings is wide (one row per prediction, one column per raw
    sensor) for easy plotting; inference_shap_values is long/normalized
    (one row per feature per prediction) so an LLM can query it with SQL
    (e.g. average SHAP per feature over a cycle range) without pivoting.
    """
    try:
        database_directory = os.path.dirname(database_path)
        if database_directory:
            os.makedirs(database_directory, exist_ok=True)
        sensor_column_definitions = ", ".join(
            f'"{column}" REAL' for column in sensor_columns
        )
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {INFERENCE_READINGS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine_id INTEGER NOT NULL,
                    cycle INTEGER NOT NULL,
                    {sensor_column_definitions},
                    predicted_life_ratio REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SHAP_VALUES_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine_id INTEGER NOT NULL,
                    cycle INTEGER NOT NULL,
                    feature_name TEXT NOT NULL,
                    shap_value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
        logging.info(f"Initialized inference log database at {database_path}")
    except Exception as error:
        raise CustomException(str(error)) from error


def log_inference_reading(
    database_path: str,
    engine_id: int,
    cycle: int,
    sensor_readings: dict[str, float],
    predicted_life_ratio: float,
) -> None:
    """Insert one raw-sensor-reading + prediction row into the inference log."""
    try:
        timestamp = datetime.now(UTC).isoformat()
        columns = [
            "engine_id",
            "cycle",
            *sensor_readings.keys(),
            "predicted_life_ratio",
            "timestamp",
        ]
        # float() guards against numpy scalars, which sqlite3 would store as BLOBs.
        values = [
            engine_id,
            cycle,
            *(float(value) for value in sensor_readings.values()),
            float(predicted_life_ratio),
            timestamp,
        ]
        column_list = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in values)
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                f"INSERT INTO {INFERENCE_READINGS_TABLE} ({column_list}) "
                f"VALUES ({placeholders})",
                values,
            )
    except Exception as error:
        raise CustomException(str(error)) from error


def log_shap_values(
    database_path: str,
    engine_id: int,
    cycle: int,
    shap_values: dict[str, float],
) -> None:
    """Insert one row per feature's SHAP value for a single prediction."""
    try:
        timestamp = datetime.now(UTC).isoformat()
        # float() guards against numpy scalars, which sqlite3 would store as BLOBs.
        rows = [
            (engine_id, cycle, feature_name, float(shap_value), timestamp)
            for feature_name, shap_value in shap_values.items()
        ]
        with sqlite3.connect(database_path) as connection:
            connection.executemany(
                f"INSERT INTO {SHAP_VALUES_TABLE} "
                "(engine_id, cycle, feature_name, shap_value, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    except Exception as error:
        raise CustomException(str(error)) from error


def delete_engine_history(database_path: str, engine_id: int) -> None:
    """Clear one engine's logged readings and SHAP values.

    Rows accumulate across every run (restarts included) with no natural
    upper bound on cycle number, so "latest cycle" queries would otherwise
    keep returning a prior run's furthest cycle instead of the current run's
    -- this is what a UI "reset" should call to give an engine a clean slate.
    """
    try:
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                f"DELETE FROM {INFERENCE_READINGS_TABLE} WHERE engine_id = ?",
                (engine_id,),
            )
            connection.execute(
                f"DELETE FROM {SHAP_VALUES_TABLE} WHERE engine_id = ?",
                (engine_id,),
            )
        logging.info(f"Cleared inference log history for engine {engine_id}")
    except Exception as error:
        raise CustomException(str(error)) from error
