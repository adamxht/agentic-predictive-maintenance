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
        explainer = shap.TreeExplainer(model)
        return explainer(sample_features)
    except Exception as error:
        raise CustomException(str(error)) from error
