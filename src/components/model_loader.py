import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

MLFLOW_URI_PREFIXES = ("models:/", "runs:/")


def load_model_for_evaluation(model_reference: str, mlflow_tracking_uri: str) -> object:
    """Load a trained model from either an MLflow URI or a local model.pkl path."""
    try:
        if model_reference.startswith(MLFLOW_URI_PREFIXES):
            return _load_from_mlflow(model_reference, mlflow_tracking_uri)
        return _load_from_local_path(model_reference)
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
