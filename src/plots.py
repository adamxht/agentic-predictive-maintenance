import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.exception import CustomException


def _save_and_close(figure: plt.Figure, output_path: str) -> None:
    """Save a matplotlib figure to disk and close it to free memory."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_actual_vs_predicted(
    true_values, predicted_values, title: str, output_path: str, sample_size: int = 200
) -> None:
    """Plot actual vs predicted values for the first sample_size rows."""
    try:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(np.asarray(true_values)[:sample_size], label="Actual")
        axis.plot(np.asarray(predicted_values)[:sample_size], label="Predicted")
        axis.set_title(title)
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Target value")
        axis.legend()
        axis.grid(True)
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_train_validation_error_curve(
    trial_history: list[dict], output_path: str
) -> None:
    """Plot train/validation RMSE across Optuna trials."""
    try:
        trial_numbers = [entry["trial"] for entry in trial_history]
        train_rmse_values = [entry["train_rmse"] for entry in trial_history]
        validation_rmse_values = [entry["validation_rmse"] for entry in trial_history]

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(trial_numbers, train_rmse_values, label="Train RMSE")
        axis.plot(trial_numbers, validation_rmse_values, label="Validation RMSE")
        axis.set_title("Train vs Validation RMSE per Optuna Trial")
        axis.set_xlabel("Trial")
        axis.set_ylabel("RMSE")
        axis.legend()
        axis.grid(True)
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_shap_beeswarm(shap_values: shap.Explanation, output_path: str) -> None:
    """Save a SHAP beeswarm summary plot."""
    try:
        shap.plots.beeswarm(shap_values, show=False)
        _save_and_close(plt.gcf(), output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_shap_bar(shap_values: shap.Explanation, output_path: str) -> None:
    """Save a SHAP bar (mean absolute SHAP value) plot."""
    try:
        shap.plots.bar(shap_values, show=False)
        _save_and_close(plt.gcf(), output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_residuals(
    true_values, predicted_values, target_label: str, output_path: str
) -> None:
    """Plot residuals (true - predicted) against the true value."""
    try:
        residuals = np.asarray(true_values) - np.asarray(predicted_values)
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.scatter(true_values, residuals, alpha=0.5)
        axis.axhline(0, color="red", linestyle="--")
        axis.set_xlabel(f"True {target_label}")
        axis.set_ylabel("Residual (True - Predicted)")
        axis.set_title(f"Residuals vs True {target_label}")
        axis.grid(True)
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_error_by_cycle(cycle_values, residuals, output_path: str) -> None:
    """Plot absolute error against cycle number."""
    try:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.scatter(cycle_values, np.abs(residuals), alpha=0.5)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Absolute Error")
        axis.set_title("Absolute Error vs Cycle")
        axis.grid(True)
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_error_by_engine(
    engine_ids, residuals, output_path: str, top_n: int = 20
) -> None:
    """Plot mean absolute error per engine, showing the top_n worst engines."""
    try:
        error_by_engine = (
            pd.DataFrame({"engine_id": engine_ids, "abs_error": np.abs(residuals)})
            .groupby("engine_id")["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )
        figure, axis = plt.subplots(figsize=(10, 4))
        error_by_engine.head(top_n).plot(kind="bar", ax=axis)
        axis.set_title("Mean Absolute Error by Engine")
        axis.set_ylabel("MAE")
        axis.set_xlabel("Engine ID")
        axis.grid(axis="y")
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_confusion_matrix(
    confusion_matrix_array: np.ndarray, class_names: list[str], output_path: str
) -> None:
    """Plot a confusion matrix heatmap."""
    try:
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            confusion_matrix_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Confusion Matrix for Near-Failure Classification")
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error


def plot_roc_curve(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    roc_auc_value: float,
    output_path: str,
) -> None:
    """Plot an ROC curve with the AUC value in the legend."""
    try:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC (AUC = {roc_auc_value:.4f})",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("ROC Curve: Near-Failure Classification")
        axis.legend()
        axis.grid(True)
        _save_and_close(figure, output_path)
    except Exception as error:
        raise CustomException(str(error)) from error
