import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd

from src import plots
from src.components import evaluate, explain, model_trainer
from src.configs.model_training_config_schema import ModelConfig, ModelTrainingConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import copy_directory_contents, save_object

PREPROCESSOR_DIR_NAME = "preprocessor"


@dataclass
class ModelRunResult:
    """Outputs produced by training and evaluating a single model."""

    model_name: str
    model: object
    train_metrics: dict
    validation_metrics: dict
    plots_directory: str
    model_path: str | None = None


@dataclass
class _ModelEvaluationContext:
    """Bundles the data needed to plot and log results for one trained model."""

    training_target: pd.Series
    train_predictions: object
    validation_features: pd.DataFrame
    validation_dataframe: pd.DataFrame
    validation_target: pd.Series
    validation_predictions: object
    trial_history: list[dict]
    roc_auc: float


class TrainingPipeline:
    """Runs hyperparameter search, evaluation, explainability, and MLflow logging."""

    def __init__(self, configuration: ModelTrainingConfig) -> None:
        self.configuration = configuration
        self.target_column = configuration.target.column_name
        self.train_dataframe: pd.DataFrame | None = None
        self.validation_dataframe: pd.DataFrame | None = None

    def run(self) -> list[ModelRunResult]:
        """Train, evaluate, and log every model configured for this run."""
        try:
            self.train_dataframe = pd.read_csv(
                self.configuration.data.processed_train_path
            )
            self.validation_dataframe = pd.read_csv(
                self.configuration.data.processed_validation_path
            )

            mlflow.set_tracking_uri(self.configuration.mlflow.tracking_uri)
            mlflow.set_experiment(self.configuration.mlflow.experiment_name)

            return [
                self._run_single_model(model_config)
                for model_config in self.configuration.models
            ]
        except Exception as error:
            logging.error(f"Model training pipeline failed: {error}")
            raise CustomException(str(error)) from error

    def _select_features_and_target(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a processed dataframe into feature matrix and target vector.

        Unlike the data preparation pipeline, "cycle" is kept as a feature here
        (only the target and engine_id are dropped), matching the notebook.
        """
        excluded_columns = {self.target_column, "engine_id"}
        feature_columns = [
            column for column in dataframe.columns if column not in excluded_columns
        ]
        return dataframe[feature_columns], dataframe[self.target_column]

    def _run_single_model(self, model_config: ModelConfig) -> ModelRunResult:
        """Train, evaluate, explain, plot, and log a single configured model."""
        logging.info(f"Training model: {model_config.name}")
        training_features, training_target = self._select_features_and_target(
            self.train_dataframe
        )
        validation_features, validation_target = self._select_features_and_target(
            self.validation_dataframe
        )

        model, trial_history = model_trainer.train_with_hyperparameter_search(
            model_config,
            training_features,
            training_target,
            validation_features,
            validation_target,
        )
        train_predictions = model.predict(training_features)
        validation_predictions = model.predict(validation_features)

        train_metrics, validation_metrics = self._compute_metrics(
            training_target,
            train_predictions,
            validation_target,
            validation_predictions,
        )
        context = _ModelEvaluationContext(
            training_target=training_target,
            train_predictions=train_predictions,
            validation_features=validation_features,
            validation_dataframe=self.validation_dataframe,
            validation_target=validation_target,
            validation_predictions=validation_predictions,
            trial_history=trial_history,
            roc_auc=validation_metrics["roc_auc"],
        )

        plots_directory = self._plots_directory_for(model_config.name)
        if self.configuration.plots.enabled:
            self._generate_plots(model, context, plots_directory)

        self._log_to_mlflow(
            model_config, model, train_metrics, validation_metrics, plots_directory
        )
        model_path = self._save_model_locally(model, model_config)

        return ModelRunResult(
            model_name=model_config.name,
            model=model,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            plots_directory=plots_directory,
            model_path=model_path,
        )

    def _save_model_locally(self, model, model_config: ModelConfig) -> str | None:
        """Optionally persist the fitted model locally, decoupled from MLflow.

        The preprocessing artifacts are copied alongside the model under a
        `preprocessor/` subfolder, so the model folder is a single,
        self-contained artifact.
        """
        if not model_config.save_locally:
            return None
        model_path = os.path.join(model_config.save_model_path, "model.pkl")
        save_object(model_path, model)
        copy_directory_contents(
            self.configuration.data.artifacts_path,
            os.path.join(model_config.save_model_path, PREPROCESSOR_DIR_NAME),
        )
        return model_path

    def _compute_metrics(
        self,
        training_target,
        train_predictions,
        validation_target,
        validation_predictions,
    ) -> tuple[dict, dict]:
        """Compute regression and binary-classification metrics for train/validation."""
        train_metrics = evaluate.compute_regression_metrics(
            training_target, train_predictions
        )
        validation_metrics = evaluate.compute_regression_metrics(
            validation_target, validation_predictions
        )

        binary_config = self.configuration.binary_classification
        validation_metrics.update(
            evaluate.compute_binary_classification_metrics(
                validation_target,
                validation_predictions,
                binary_config.threshold,
                binary_config.pred_offset,
            )
        )
        return train_metrics, validation_metrics

    def _plots_directory_for(self, model_name: str) -> str:
        """Return the plots output directory for a given model within this run."""
        return os.path.join(
            self.configuration.plots.output_dir,
            self.configuration.run_name,
            model_name,
            "plots",
        )

    def _generate_plots(
        self, model, context: _ModelEvaluationContext, plots_directory: str
    ) -> None:
        """Generate and save all evaluation/explainability plots for a trained model."""
        plots.plot_actual_vs_predicted(
            context.training_target,
            context.train_predictions,
            "Train: Actual vs Predicted",
            os.path.join(plots_directory, "actual_vs_predicted_train.png"),
        )
        plots.plot_actual_vs_predicted(
            context.validation_target,
            context.validation_predictions,
            "Validation: Actual vs Predicted",
            os.path.join(plots_directory, "actual_vs_predicted_val.png"),
        )
        plots.plot_train_validation_error_curve(
            context.trial_history,
            os.path.join(plots_directory, "train_val_error_curve.png"),
        )
        self._generate_shap_plots(model, context.validation_features, plots_directory)
        self._generate_error_analysis_plots(context, plots_directory)
        self._generate_binary_classification_plots(context, plots_directory)

    def _generate_shap_plots(
        self, model, validation_features: pd.DataFrame, plots_directory: str
    ) -> None:
        """Compute and plot SHAP values for a sample of validation rows."""
        explain_config = self.configuration.explainability
        sample_features = explain.sample_features_for_explanation(
            validation_features, explain_config.sample_size, explain_config.random_state
        )
        shap_values = explain.compute_shap_values(model, sample_features)
        plots.plot_shap_beeswarm(
            shap_values, os.path.join(plots_directory, "shap_beeswarm.png")
        )
        plots.plot_shap_bar(shap_values, os.path.join(plots_directory, "shap_bar.png"))

    def _generate_error_analysis_plots(
        self, context: _ModelEvaluationContext, plots_directory: str
    ) -> None:
        """Plot residuals, error-by-cycle, and error-by-engine for validation data."""
        residuals = (
            context.validation_target.to_numpy() - context.validation_predictions
        )
        plots.plot_residuals(
            context.validation_target,
            context.validation_predictions,
            self.target_column,
            os.path.join(plots_directory, "residuals.png"),
        )
        plots.plot_error_by_cycle(
            context.validation_dataframe["cycle"],
            residuals,
            os.path.join(plots_directory, "error_by_cycle.png"),
        )
        plots.plot_error_by_engine(
            context.validation_dataframe["engine_id"],
            residuals,
            os.path.join(plots_directory, "error_by_engine.png"),
        )

    def _generate_binary_classification_plots(
        self, context: _ModelEvaluationContext, plots_directory: str
    ) -> None:
        """Plot the near-failure confusion matrix and ROC curve for validation data."""
        binary_config = self.configuration.binary_classification
        confusion_matrix_array = evaluate.compute_confusion_matrix(
            context.validation_target,
            context.validation_predictions,
            binary_config.threshold,
            binary_config.pred_offset,
        )
        plots.plot_confusion_matrix(
            confusion_matrix_array,
            ["Not Near Failure", "Near Failure"],
            os.path.join(plots_directory, "confusion_matrix.png"),
        )

        false_positive_rate, true_positive_rate, _ = evaluate.compute_roc_curve(
            context.validation_target,
            context.validation_predictions,
            binary_config.threshold,
        )
        plots.plot_roc_curve(
            false_positive_rate,
            true_positive_rate,
            context.roc_auc,
            os.path.join(plots_directory, "roc_curve.png"),
        )

    def _log_to_mlflow(
        self,
        model_config: ModelConfig,
        model,
        train_metrics: dict,
        validation_metrics: dict,
        plots_directory: str,
    ) -> None:
        """Log this model's config, metrics, lineage, and artifacts to MLflow."""
        run_name = f"{self.configuration.run_name}_{model_config.name}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_config.name)
            mlflow.log_param("target_type", self.configuration.target.type)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(
                {f"train_{name}": value for name, value in train_metrics.items()}
            )
            mlflow.log_metrics(
                {
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                }
            )

            self._log_datasets_to_mlflow()

            if self.configuration.plots.enabled and os.path.isdir(plots_directory):
                mlflow.log_artifacts(plots_directory, artifact_path="plots")

            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=model_config.registered_model_name,
            )
            self._log_preprocessor_to_mlflow()

    def _log_preprocessor_to_mlflow(self) -> None:
        """Log every preprocessing artifact nested under the model artifact.

        Places them at model/preprocessor/, so the logged model artifact is
        self-contained the same way a locally saved model folder is.
        """
        mlflow.log_artifacts(
            self.configuration.data.artifacts_path,
            artifact_path=os.path.join("model", PREPROCESSOR_DIR_NAME),
        )

    def _log_datasets_to_mlflow(self) -> None:
        """Log the train/validation dataset lineage (source, schema) to this run."""
        train_dataset = mlflow.data.from_pandas(
            self.train_dataframe,
            source=self.configuration.data.processed_train_path,
            name="train",
            targets=self.target_column,
        )
        validation_dataset = mlflow.data.from_pandas(
            self.validation_dataframe,
            source=self.configuration.data.processed_validation_path,
            name="validation",
            targets=self.target_column,
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(validation_dataset, context="validation")
