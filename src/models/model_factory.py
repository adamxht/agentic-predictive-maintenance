from typing import ClassVar

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.exception import CustomException


class ModelFactory:
    """Builds a regressor instance for a configured model name."""

    _BUILDERS: ClassVar[dict[str, type]] = {
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
    }

    @classmethod
    def create(cls, model_name: str, params: dict) -> object:
        """Instantiate the registered model for model_name with the given parameters."""
        try:
            builder = cls._BUILDERS[model_name]
        except KeyError as error:
            raise CustomException(f"Unknown model name: {model_name}") from error
        return builder(**params)
