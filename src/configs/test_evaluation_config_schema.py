from typing import Literal

import yaml
from pydantic import BaseModel

from src.exception import CustomException


class TestEvaluationOverridesConfig(BaseModel):
    """Optional overrides for run_test_set_eval.py.

    Every field defaults to None (meaning "not specified"). Any field that IS
    set here overwrites the corresponding CLI argument, per the project's
    config-overrides-CLI convention for this script.
    """

    model: str | None = None
    mlflow_tracking_uri: str | None = None
    target_type: Literal["rul", "life_ratio"] | None = None
    threshold: float | None = None
    pred_offset: float | None = None
    sample_size: int | None = None
    explain_random_state: int | None = None
    plots_enabled: bool | None = None
    plots_output_dir: str | None = None


def load_test_evaluation_overrides(config_path: str) -> TestEvaluationOverridesConfig:
    """Load and validate test-evaluation config overrides from a YAML file."""
    try:
        with open(config_path) as config_file:
            raw_configuration = yaml.safe_load(config_file) or {}
        return TestEvaluationOverridesConfig(**raw_configuration)
    except Exception as error:
        raise CustomException(str(error)) from error
