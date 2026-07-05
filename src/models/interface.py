from abc import ABC, abstractmethod
from typing import Any, Self

# Some models, such as PyTorch models, do not implement the standard scikit-learn API (e.g., .fit(), .predict()). We wrap or subclass these models to provide a scikit-learn-compatible interface.
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