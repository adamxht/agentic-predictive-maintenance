import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.exception import CustomException


def compute_regression_metrics(true_values, predicted_values) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 for a regression prediction."""
    try:
        return {
            "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
            "mae": float(mean_absolute_error(true_values, predicted_values)),
            "r2": float(r2_score(true_values, predicted_values)),
        }
    except Exception as error:
        raise CustomException(str(error)) from error


def derive_near_failure_labels(
    true_values, predicted_values, threshold: float, pred_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold continuous true/predicted values into near-failure binary labels.

    A lower value (life_ratio or RUL) means closer to failure, so both labels
    are derived as "value <= threshold".
    """
    try:
        true_labels = (np.asarray(true_values) <= threshold).astype(int)
        predicted_labels = (
            np.asarray(predicted_values) <= threshold + pred_offset
        ).astype(int)
        return true_labels, predicted_labels
    except Exception as error:
        raise CustomException(str(error)) from error


def compute_binary_classification_metrics(
    true_values, predicted_values, threshold: float, pred_offset: float = 0.0
) -> dict[str, float]:
    """Compute accuracy, precision, recall, f1, and roc-auc for near-failure classes."""
    try:
        true_labels, predicted_labels = derive_near_failure_labels(
            true_values, predicted_values, threshold, pred_offset
        )
        predicted_risk_scores = -np.asarray(predicted_values)

        metrics = {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "precision": float(
                precision_score(true_labels, predicted_labels, zero_division=0)
            ),
            "recall": float(
                recall_score(true_labels, predicted_labels, zero_division=0)
            ),
            "f1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
        }
        metrics["roc_auc"] = (
            float(roc_auc_score(true_labels, predicted_risk_scores))
            if len(np.unique(true_labels)) > 1
            else float("nan")
        )
        return metrics
    except Exception as error:
        raise CustomException(str(error)) from error


def compute_confusion_matrix(
    true_values, predicted_values, threshold: float, pred_offset: float = 0.0
) -> np.ndarray:
    """Compute the near-failure confusion matrix for continuous predictions."""
    try:
        true_labels, predicted_labels = derive_near_failure_labels(
            true_values, predicted_values, threshold, pred_offset
        )
        return confusion_matrix(true_labels, predicted_labels)
    except Exception as error:
        raise CustomException(str(error)) from error


def compute_roc_curve(
    true_values, predicted_values, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve points: false positive rate, true positive rate, thresholds."""
    try:
        true_labels = (np.asarray(true_values) <= threshold).astype(int)
        predicted_risk_scores = -np.asarray(predicted_values)
        return roc_curve(true_labels, predicted_risk_scores)
    except Exception as error:
        raise CustomException(str(error)) from error
