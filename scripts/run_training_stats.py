"""Entry point that precomputes per-sensor training statistics for the drift tool."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.components.data_ingestion import load_raw_sensor_readings
from src.const import ENGINE_ID_COLUMN, SENSOR_NAMES
from src.exception import CustomException
from src.logger import logging

DEFAULT_TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "configs" / "agent" / "training_statistics.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the training statistics entry point."""
    parser = argparse.ArgumentParser(
        description="Precompute per-sensor training statistics (mean/std/min/max) "
        "used by the MCP drift tool."
    )
    parser.add_argument("--train-data-path", default=str(DEFAULT_TRAIN_DATA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def compute_sensor_statistics(raw_readings_df: pd.DataFrame) -> dict:
    """Compute mean/std/min/max per raw sensor column of the training data."""
    try:
        statistics = {}
        for sensor_name in SENSOR_NAMES:
            sensor_values = raw_readings_df[sensor_name].astype(float)
            statistics[sensor_name] = {
                "mean": float(sensor_values.mean()),
                "std": float(sensor_values.std(ddof=0)),
                "min": float(sensor_values.min()),
                "max": float(sensor_values.max()),
            }
        return statistics
    except Exception as error:
        raise CustomException(str(error)) from error


def write_statistics_document(
    statistics: dict, raw_readings_df: pd.DataFrame, source_path: str, output_path: str
) -> None:
    """Write the statistics plus dataset lineage metadata as formatted JSON."""
    try:
        if os.path.isabs(source_path):
            source_path = os.path.relpath(source_path, PROJECT_ROOT)
        document = {
            "source_file": source_path,
            "row_count": len(raw_readings_df),
            "engine_count": int(raw_readings_df[ENGINE_ID_COLUMN].nunique()),
            "statistics": statistics,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(document, output_file, indent=2)
            output_file.write("\n")
    except Exception as error:
        raise CustomException(str(error)) from error


def main() -> None:
    """Compute and save the training statistics artifact."""
    arguments = parse_arguments()
    raw_readings_df = load_raw_sensor_readings(arguments.train_data_path)
    statistics = compute_sensor_statistics(raw_readings_df)
    write_statistics_document(
        statistics, raw_readings_df, arguments.train_data_path, arguments.output_path
    )
    logging.info(
        f"Wrote statistics for {len(statistics)} sensors "
        f"({len(raw_readings_df)} rows) to {arguments.output_path}"
    )


if __name__ == "__main__":
    main()
