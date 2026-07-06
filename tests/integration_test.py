from pathlib import Path

import pytest

from src.components import evaluate
from src.configs.data_pipeline_config_schema import load_data_preparation_config
from src.models.model_factory import ModelFactory
from src.pipeline.data_preparation_pipeline import (
    DataPreparationPipeline,
    TestSetPreparationPipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data_transformation" / "default.yaml"

RANDOM_FOREST_BEST_PARAMS = {
    "n_estimators": 447,
    "max_depth": 10,
    "min_samples_split": 17,
    "min_samples_leaf": 8,
    "max_features": 0.5,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_BEST_PARAMS = {
    "n_estimators": 369,
    "max_depth": 5,
    "learning_rate": 0.15465998746633278,
    "subsample": 0.6135451690005006,
    "colsample_bytree": 0.7455306127714851,
    "min_child_weight": 9,
    "gamma": 0.04919857761786995,
    "reg_alpha": 0.6748439844247618,
    "reg_lambda": 0.41907927448580273,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}


def _load_data_preparation_config_with_tmp_outputs(tmp_path: Path):
    """Load the real data-prep config, redirecting outputs to a temp directory.

    Raw input paths are left as-is (the real data/raw/*.txt), so this
    exercises the actual pipelines against real data without ever writing to
    the project's real data/processed/ directory.
    """
    configuration = load_data_preparation_config(str(DATA_CONFIG_PATH))
    configuration.paths.processed_train_path = str(tmp_path / "train.csv")
    configuration.paths.processed_validation_path = str(tmp_path / "val.csv")
    configuration.paths.scaler_path = str(tmp_path / "scaler.pkl")
    configuration.paths.selected_features_path = str(
        tmp_path / "selected_features.json"
    )
    configuration.test_set.processed_test_path = str(tmp_path / "test.csv")
    return configuration


def _split_features_and_target(dataframe, target_column: str):
    """Match the training pipeline: drop only the target and engine_id, keep cycle."""
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {target_column, "engine_id"}
    ]
    return dataframe[feature_columns], dataframe[target_column]


@pytest.fixture(scope="module")
def prepared_real_datasets(tmp_path_factory) -> dict:
    """Run the real data-prep and test-set pipelines against the real CMAPSS data."""
    tmp_path = tmp_path_factory.mktemp("integration_data")
    configuration = _load_data_preparation_config_with_tmp_outputs(tmp_path)
    target_column = configuration.target.column_name

    data_preparation_artifacts = DataPreparationPipeline(configuration).run()
    test_preparation_artifacts = TestSetPreparationPipeline(configuration).run()

    training_features, training_target = _split_features_and_target(
        data_preparation_artifacts.train_dataframe, target_column
    )
    validation_features, validation_target = _split_features_and_target(
        data_preparation_artifacts.validation_dataframe, target_column
    )
    test_features, test_target = _split_features_and_target(
        test_preparation_artifacts.test_dataframe, target_column
    )

    return {
        "training_features": training_features,
        "training_target": training_target,
        "validation_features": validation_features,
        "validation_target": validation_target,
        "test_features": test_features,
        "test_target": test_target,
    }


def _fit_and_evaluate(
    model_name: str, params: dict, datasets: dict
) -> tuple[dict, dict]:
    """Fit a model with fixed hyperparameters and evaluate on validation/test data."""
    model = ModelFactory.create(model_name, params)
    model.fit(datasets["training_features"], datasets["training_target"])

    validation_predictions = model.predict(datasets["validation_features"])
    test_predictions = model.predict(datasets["test_features"])

    validation_metrics = evaluate.compute_regression_metrics(
        datasets["validation_target"], validation_predictions
    )
    test_metrics = evaluate.compute_regression_metrics(
        datasets["test_target"], test_predictions
    )
    return validation_metrics, test_metrics


def test_random_forest_pipeline_matches_baseline(prepared_real_datasets):
    """Regression-detect the full data-prep -> train -> eval flow for RandomForest.

    Baseline captured by running this suite once against the hardcoded
    RANDOM_FOREST_BEST_PARAMS on the real CMAPSS data.
    """
    validation_metrics, test_metrics = _fit_and_evaluate(
        "random_forest", RANDOM_FOREST_BEST_PARAMS, prepared_real_datasets
    )

    # Hard coded baselines
    assert validation_metrics["rmse"] == pytest.approx(0.0571, rel=0.15)
    assert validation_metrics["r2"] == pytest.approx(0.960, abs=0.05)
    assert test_metrics["rmse"] == pytest.approx(0.0668, rel=0.15)
    assert test_metrics["r2"] == pytest.approx(0.914, abs=0.05)


def test_xgboost_pipeline_matches_baseline(prepared_real_datasets):
    """Regression-detect the full data-prep -> train -> eval flow for XGBoost.

    Baseline captured by running this suite once against the hardcoded
    XGBOOST_BEST_PARAMS on the real CMAPSS data.
    """
    validation_metrics, test_metrics = _fit_and_evaluate(
        "xgboost", XGBOOST_BEST_PARAMS, prepared_real_datasets
    )

    # Hard coded baselines
    assert validation_metrics["rmse"] == pytest.approx(0.0568, rel=0.15)
    assert validation_metrics["r2"] == pytest.approx(0.960, abs=0.05)
    assert test_metrics["rmse"] == pytest.approx(0.0667, rel=0.15)
    assert test_metrics["r2"] == pytest.approx(0.914, abs=0.05)
