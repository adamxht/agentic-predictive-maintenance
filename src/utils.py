import json
import os
import shutil

import joblib
import pandas as pd
from tabulate import tabulate

from src.const import NON_FEATURE_COLUMNS
from src.exception import CustomException
from src.logger import logging


def get_sensor_columns(
    dataframe: pd.DataFrame, excluded_columns: set[str] | None = None
) -> list[str]:
    """Return numeric sensor columns from a dataframe, excluding identifier/target."""
    excluded = excluded_columns if excluded_columns is not None else NON_FEATURE_COLUMNS
    return [
        column
        for column in dataframe.select_dtypes(include=["number"]).columns
        if column not in excluded
    ]


def format_metrics_table(metrics: dict[str, float]) -> str:
    """Format a metrics dict as a human-readable table for console/log output."""
    try:
        rows = [(name.upper(), f"{value:.4f}") for name, value in metrics.items()]
        return tabulate(rows, headers=["Metric", "Value"], tablefmt="fancy_grid")
    except Exception as error:
        raise CustomException(str(error)) from error


def save_dataframe(dataframe: pd.DataFrame, file_path: str) -> None:
    """Save a dataframe to CSV, creating parent directories if needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.to_csv(file_path, index=False)
        logging.info(f"Saved dataframe with shape {dataframe.shape} to {file_path}")
    except Exception as error:
        raise CustomException(str(error)) from error


def save_object(file_path: str, object_to_save: object) -> None:
    """Persist a Python object to disk using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(object_to_save, file_path)
        logging.info(f"Saved object to {file_path}")
    except Exception as error:
        raise CustomException(str(error)) from error


def load_object(file_path: str) -> object:
    """Load a Python object previously saved with save_object."""
    try:
        return joblib.load(file_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def copy_directory_contents(source_directory: str, destination_directory: str) -> None:
    """Copy every file in source_directory into destination_directory."""
    try:
        shutil.copytree(source_directory, destination_directory, dirs_exist_ok=True)
        logging.info(f"Copied {source_directory} to {destination_directory}")
    except Exception as error:
        raise CustomException(str(error)) from error


def save_json(data: object, file_path: str) -> None:
    """Save a JSON-serializable object to disk, creating parent directories."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=2)
        logging.info(f"Saved JSON to {file_path}")
    except Exception as error:
        raise CustomException(str(error)) from error


def load_json(file_path: str) -> object:
    """Load a JSON object previously saved with save_json."""
    try:
        with open(file_path) as json_file:
            return json.load(json_file)
    except Exception as error:
        raise CustomException(str(error)) from error


def safe_downcast_with_check(df, datatype):
    """
    Safely downcast selected columns and print min/max validation report.
    """

    df_original = df.copy()

    # select columns
    cols = df.select_dtypes(include=[datatype]).columns

    print("\n=== SAFE DOWNSCAST ===\n")

    for col in cols:
        col_data = df[col]

        before_min = col_data.min()
        before_max = col_data.max()
        before_dtype = col_data.dtype

        # downcast attempt
        if pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
        elif pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")

        after_min = df[col].min()
        after_max = df[col].max()
        after_dtype = df[col].dtype

        safe = (
            (pd.isna(before_min) and pd.isna(after_min)) or before_min == after_min
        ) and ((pd.isna(before_max) and pd.isna(after_max)) or before_max == after_max)

        if not safe:
            df[col] = df_original[col]
            status = "ROLLBACK (min/max changed)"
        else:
            status = f"DOWNCASTED → {after_dtype}"

        print(f"\nColumn: {col}")
        print(f"  Before dtype: {before_dtype} | After dtype: {df[col].dtype}")
        print(f"  Min: {before_min} → {df[col].min()}")
        print(f"  Max: {before_max} → {df[col].max()}")
        print(f"  Status: {status}")

    return df
