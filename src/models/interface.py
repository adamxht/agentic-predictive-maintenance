from abc import ABC, abstractmethod
from typing import Any, Self


# Some models (e.g. PyTorch) don't implement fit()/predict() natively, so we wrap them.
class SklearnModelInterface(ABC):
    """Interface for models compatible with the scikit-learn API."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> Self:
        """Train the model and return the fitted instance."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions for the input data."""
        raise NotImplementedError
