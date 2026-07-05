"""Entry point that runs the data preparation pipeline from a YAML config."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config_schema import load_data_preparation_config
from src.exception import CustomException
from src.logger import logging
from src.pipeline.data_preparation_pipeline import DataPreparationPipeline

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_transformation" / "default.yaml"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the data preparation entry point."""
    parser = argparse.ArgumentParser(
        description="Run the CMAPSS data preparation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a data preparation YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configured pipeline and run it end to end."""
    arguments = parse_arguments()
    logging.info(f"Loading data preparation config from {arguments.config}")
    configuration = load_data_preparation_config(arguments.config)

    pipeline = DataPreparationPipeline(configuration)
    artifacts = pipeline.run()

    logging.info(
        f"Data preparation complete. Train shape: {artifacts.train_dataframe.shape}, "
        f"Validation shape: {artifacts.validation_dataframe.shape}"
    )


if __name__ == "__main__":
    try:
        main()
    except CustomException as error:
        logging.error(str(error))
        sys.exit(1)
