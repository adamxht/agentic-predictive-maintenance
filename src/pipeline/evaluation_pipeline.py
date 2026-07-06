import os
from dataclasses import dataclass

import pandas as pd

from src import plots
from src.components import evaluate, explain, model_loader
from src.configs.data_pipeline_config_schema import TARGET_COLUMN_BY_TYPE
from src.exception import CustomException
from src.logger import logging
from src.utils import format_metrics_table


@dataclass
class TestSetEvaluationSettings:
    """Resolved settings for one test-set evaluation run (CLI args + config merged)."""

    test_data_path: str
    model_reference: str
    mlflow_tracking_uri: str
    target_type: str
    threshold: float
    pred_offset: float
    sample_size: int
    explain_random_state: int
    plots_enabled: bool
    plots_output_dir: str


@dataclass
class TestSetEvaluationResult:
    """Outputs produced by evaluating a model against the held-out test set."""

    metrics: dict
    plots_directory: str


class TestSetEvaluationPipeline:
    """Loads a trained model and evaluates it against the processed test set."""

    def __init__(self, settings: TestSetEvaluationSettings) -> None:
        self.settings = settings
        self.target_column = TARGET_COLUMN_BY_TYPE[settings.target_type]

    def run(self) -> TestSetEvaluationResult:
        """Load the model and test data, evaluate, plot, and return the metrics."""
        try:
            test_dataframe = pd.read_csv(self.settings.test_data_path)
            model = model_loader.load_model_for_evaluation(
                self.settings.model_reference, self.settings.mlflow_tracking_uri
            )
            test_features, test_target = self._select_features_and_target(
                test_dataframe
            )
            test_predictions = model.predict(test_features)

            metrics = self._compute_metrics(test_target, test_predictions)
            model_identifier = model_loader.derive_model_identifier(
                self.settings.model_reference
            )
            plots_directory = os.path.join(
                self.settings.plots_output_dir, model_identifier, "plots"
            )
            if self.settings.plots_enabled:
                self._generate_plots(
                    model,
                    test_dataframe,
                    test_features,
                    test_target,
                    test_predictions,
                    metrics["roc_auc"],
                    plots_directory,
                )

            logging.info(
                f"Test set evaluation metrics:\n{format_metrics_table(metrics)}"
            )
            return TestSetEvaluationResult(
                metrics=metrics, plots_directory=plots_directory
            )
        except Exception as error:
            logging.error(f"Test set evaluation failed: {error}")
            raise CustomException(str(error)) from error

    def _select_features_and_target(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a processed dataframe into feature matrix and target vector.

        Matches the training pipeline: only the target and engine_id are
        dropped, "cycle" is kept as a feature.
        """
        excluded_columns = {self.target_column, "engine_id"}
        feature_columns = [
            column for column in dataframe.columns if column not in excluded_columns
        ]
        return dataframe[feature_columns], dataframe[self.target_column]

    def _compute_metrics(self, test_target, test_predictions) -> dict:
        """Compute regression and binary-classification metrics for the test set."""
        metrics = evaluate.compute_regression_metrics(test_target, test_predictions)
        metrics.update(
            evaluate.compute_binary_classification_metrics(
                test_target,
                test_predictions,
                self.settings.threshold,
                self.settings.pred_offset,
            )
        )
        return metrics

    def _generate_plots(
        self,
        model,
        test_dataframe: pd.DataFrame,
        test_features: pd.DataFrame,
        test_target: pd.Series,
        test_predictions,
        roc_auc: float,
        plots_directory: str,
    ) -> None:
        """Generate and save the evaluation/explainability plots for the test set."""
        plots.plot_actual_vs_predicted(
            test_target,
            test_predictions,
            "Test: Actual vs Predicted",
            os.path.join(plots_directory, "actual_vs_predicted_test.png"),
        )
        self._generate_shap_plots(model, test_features, plots_directory)
        self._generate_error_analysis_plots(
            test_dataframe, test_target, test_predictions, plots_directory
        )
        self._generate_binary_classification_plots(
            test_target, test_predictions, roc_auc, plots_directory
        )

    def _generate_shap_plots(
        self, model, test_features: pd.DataFrame, plots_directory: str
    ) -> None:
        """Compute and plot SHAP values for a sample of test rows."""
        sample_features = explain.sample_features_for_explanation(
            test_features, self.settings.sample_size, self.settings.explain_random_state
        )
        shap_values = explain.compute_shap_values(model, sample_features)
        plots.plot_shap_beeswarm(
            shap_values, os.path.join(plots_directory, "shap_beeswarm.png")
        )
        plots.plot_shap_bar(shap_values, os.path.join(plots_directory, "shap_bar.png"))

    def _generate_error_analysis_plots(
        self,
        test_dataframe: pd.DataFrame,
        test_target: pd.Series,
        test_predictions,
        plots_directory: str,
    ) -> None:
        """Plot residuals, error-by-cycle, and error-by-engine for the test set."""
        residuals = test_target.to_numpy() - test_predictions
        plots.plot_residuals(
            test_target,
            test_predictions,
            self.target_column,
            os.path.join(plots_directory, "residuals.png"),
        )
        plots.plot_error_by_cycle(
            test_dataframe["cycle"],
            residuals,
            os.path.join(plots_directory, "error_by_cycle.png"),
        )
        plots.plot_error_by_engine(
            test_dataframe["engine_id"],
            residuals,
            os.path.join(plots_directory, "error_by_engine.png"),
        )

    def _generate_binary_classification_plots(
        self, test_target, test_predictions, roc_auc: float, plots_directory: str
    ) -> None:
        """Plot the near-failure confusion matrix and ROC curve for the test set."""
        confusion_matrix_array = evaluate.compute_confusion_matrix(
            test_target,
            test_predictions,
            self.settings.threshold,
            self.settings.pred_offset,
        )
        plots.plot_confusion_matrix(
            confusion_matrix_array,
            ["Not Near Failure", "Near Failure"],
            os.path.join(plots_directory, "confusion_matrix.png"),
        )

        false_positive_rate, true_positive_rate, _ = evaluate.compute_roc_curve(
            test_target, test_predictions, self.settings.threshold
        )
        plots.plot_roc_curve(
            false_positive_rate,
            true_positive_rate,
            roc_auc,
            os.path.join(plots_directory, "roc_curve.png"),
        )
