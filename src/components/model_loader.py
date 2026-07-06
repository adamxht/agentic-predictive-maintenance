import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

MLFLOW_URI_PREFIXES = ("models:/", "runs:/")
TRAINED_MODEL_DIR_NAME = "trained_model"


def load_model_for_evaluation(model_reference: str, mlflow_tracking_uri: str) -> object:
    """Load a trained model from either an MLflow URI or a local model.pkl path."""
    try:
        if model_reference.startswith(MLFLOW_URI_PREFIXES):
            return _load_from_mlflow(model_reference, mlflow_tracking_uri)
        return _load_from_local_path(model_reference)
    except Exception as error:
        raise CustomException(str(error)) from error


def derive_model_identifier(model_reference: str) -> str:
    """Derive a filesystem-safe identifier from a model reference.

    Used to name the evaluation output directory after the model itself
    (e.g. "life_ratio_rf_xgb/random_forest" for a local trained_model/ path,
    or "life_ratio_xgboost/1" for an MLflow models:/ URI) instead of a
    generic run name.
    """
    try:
        for prefix in MLFLOW_URI_PREFIXES:
            if model_reference.startswith(prefix):
                return model_reference[len(prefix) :].rstrip("/")

        normalized_path = model_reference.rstrip("/\\")
        if os.path.isfile(normalized_path):
            normalized_path = os.path.dirname(normalized_path)

        path_parts = os.path.normpath(normalized_path).split(os.sep)
        if TRAINED_MODEL_DIR_NAME in path_parts:
            trained_model_index = path_parts.index(TRAINED_MODEL_DIR_NAME)
            return os.path.join(*path_parts[trained_model_index + 1 :])
        return os.path.join(*path_parts[-2:]) if len(path_parts) > 1 else path_parts[-1]
    except Exception as error:
        raise CustomException(str(error)) from error


def _load_from_mlflow(model_reference: str, mlflow_tracking_uri: str) -> object:
    """Load a model logged/registered in MLflow (e.g. models:/name/1, runs:/id/model)"""
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info(f"Loading model from MLflow: {model_reference}")
    return mlflow.pyfunc.load_model(model_reference)


def _load_from_local_path(model_reference: str) -> object:
    """Load a model saved via save_object, given a folder or a direct .pkl path."""
    model_file_path = model_reference
    if os.path.isdir(model_reference):
        model_file_path = os.path.join(model_reference, "model.pkl")
    logging.info(f"Loading model from local path: {model_file_path}")
    return load_object(model_file_path)
