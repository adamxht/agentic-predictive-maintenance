import pandas as pd
import shap

from src.exception import CustomException


def sample_features_for_explanation(
    features: pd.DataFrame, sample_size: int, random_state: int
) -> pd.DataFrame:
    """Sample rows from a feature matrix for SHAP explanation, capped to its length."""
    try:
        return features.sample(
            n=min(sample_size, len(features)), random_state=random_state
        )
    except Exception as error:
        raise CustomException(str(error)) from error


def compute_shap_values(model, sample_features: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a sample of feature rows using a tree explainer."""
    try:
        explainer = shap.TreeExplainer(_unwrap_native_model(model))
        return explainer(sample_features)
    except Exception as error:
        raise CustomException(str(error)) from error


def _unwrap_native_model(model) -> object:
    """Return the raw sklearn/xgboost estimator, unwrapping an MLflow pyfunc model.

    mlflow.pyfunc.load_model() returns a generic PyFuncModel wrapper so
    .predict() works uniformly across flavors, but SHAP's TreeExplainer needs
    the actual tree-based estimator underneath. Models loaded from a local
    path (joblib) are already the raw estimator, so they pass through as-is.
    """
    model_impl = getattr(model, "_model_impl", None)
    if model_impl is not None and hasattr(model_impl, "get_raw_model"):
        return model_impl.get_raw_model()
    return model
