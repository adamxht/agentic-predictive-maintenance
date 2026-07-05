"""Entry point that evaluates a trained model against the processed test set.

The model can be either an MLflow URI (models:/... or runs:/...) or a local
path to a model saved via the training pipeline's save_locally option.
Settings can be passed as CLI arguments, a --config YAML, or both -- any
field set in --config overwrites the corresponding CLI argument.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.configs.test_evaluation_config_schema import load_test_evaluation_overrides
from src.exception import CustomException
from src.logger import logging
from src.pipeline.evaluation_pipeline import (
    TestSetEvaluationPipeline,
    TestSetEvaluationSettings,
)

DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the test-set evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model against the processed CMAPSS test set."
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(DEFAULT_TEST_DATA_PATH),
        help="Path to the processed test CSV (produced by run_data_preparation.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="MLflow URI (models:/... or runs:/...) or a local path to a model.pkl.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db"
    )
    parser.add_argument(
        "--target-type", type=str, choices=["rul", "life_ratio"], default="life_ratio"
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--pred-offset", type=float, default=0.0)
    parser.add_argument("--run-name", type=str, default="test_eval")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--explain-random-state", type=int, default=42)
    parser.add_argument(
        "--no-plots", dest="plots_enabled", action="store_false", default=True
    )
    parser.add_argument("--plots-output-dir", type=str, default="test_logs")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML; any field present overrides the arguments above.",
    )
    return parser.parse_args()


def _apply_config_overrides(arguments: argparse.Namespace) -> argparse.Namespace:
    """Overlay any values set in --config onto the parsed CLI arguments."""
    if not arguments.config:
        return arguments

    overrides = load_test_evaluation_overrides(arguments.config)
    overridden_fields = [
        field_name
        for field_name, value in overrides.model_dump().items()
        if value is not None
    ]
    if overridden_fields:
        logging.warning(
            f"Config {arguments.config} overrides CLI args: {overridden_fields}"
        )

    for field_name in overridden_fields:
        setattr(arguments, field_name, getattr(overrides, field_name))
    return arguments


def _build_settings(arguments: argparse.Namespace) -> TestSetEvaluationSettings:
    """Validate the merged arguments and build the pipeline settings."""
    if not arguments.model:
        raise ValueError(
            "A model must be provided via --model or the config's `model` field."
        )
    return TestSetEvaluationSettings(
        test_data_path=arguments.test_data,
        model_reference=arguments.model,
        mlflow_tracking_uri=arguments.mlflow_tracking_uri,
        target_type=arguments.target_type,
        threshold=arguments.threshold,
        pred_offset=arguments.pred_offset,
        run_name=arguments.run_name,
        sample_size=arguments.sample_size,
        explain_random_state=arguments.explain_random_state,
        plots_enabled=arguments.plots_enabled,
        plots_output_dir=arguments.plots_output_dir,
    )


def main() -> None:
    """Load settings, run the evaluation pipeline, and log the results."""
    try:
        arguments = parse_arguments()
        arguments = _apply_config_overrides(arguments)
        settings = _build_settings(arguments)

        pipeline = TestSetEvaluationPipeline(settings)
        result = pipeline.run()

        logging.info(f"Plots saved to: {result.plots_directory}")
    except CustomException:
        raise
    except Exception as error:
        raise CustomException(str(error)) from error


if __name__ == "__main__":
    try:
        main()
    except CustomException as error:
        logging.error(str(error))
        sys.exit(1)
