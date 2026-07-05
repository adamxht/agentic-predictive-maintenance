"""Entry point that runs the model training pipeline from a YAML config."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.configs.model_training_config_schema import load_model_training_config
from src.exception import CustomException
from src.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training" / "default.yaml"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the model training entry point."""
    parser = argparse.ArgumentParser(
        description="Run the CMAPSS model training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a model training YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configured pipeline and run it end to end."""
    arguments = parse_arguments()
    logging.info(f"Loading model training config from {arguments.config}")
    configuration = load_model_training_config(arguments.config)

    pipeline = TrainingPipeline(configuration)
    results = pipeline.run()

    for result in results:
        logging.info(
            f"{result.model_name}: validation metrics {result.validation_metrics}"
        )
        logging.info(
            f"{result.model_name}: plots at {result.plots_directory}, "
            f"local model path: {result.model_path}"
        )


if __name__ == "__main__":
    try:
        main()
    except CustomException as error:
        logging.error(str(error))
        sys.exit(1)
