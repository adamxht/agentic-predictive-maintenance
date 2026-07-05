import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.model_training_config_schema import HyperparameterSpec, ModelConfig
from src.models.model_factory import ModelFactory

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_hyperparameter(trial: optuna.Trial, name: str, spec: HyperparameterSpec):
    """Suggest a single hyperparameter value from an Optuna trial per its spec type."""
    try:
        if spec.type == "int":
            return trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        if spec.type == "float":
            return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        if spec.type == "categorical":
            return trial.suggest_categorical(name, spec.choices)
        raise ValueError(f"Unsupported hyperparameter type: {spec.type}")
    except Exception as error:
        raise CustomException(str(error)) from error


def suggest_hyperparameters(
    trial: optuna.Trial, search_space: dict[str, HyperparameterSpec]
) -> dict:
    """Suggest a full set of hyperparameters from an Optuna trial."""
    return {
        name: suggest_hyperparameter(trial, name, spec)
        for name, spec in search_space.items()
    }


def train_with_hyperparameter_search(
    model_config: ModelConfig,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> tuple[object, list[dict]]:
    """Run an Optuna study for one model, returning the best model and RMSE history."""
    try:
        trial_history = []
        best_state = {"model": None, "validation_rmse": None}

        def objective(trial: optuna.Trial) -> float:
            hyperparameters = suggest_hyperparameters(trial, model_config.search_space)
            model = ModelFactory.create(
                model_config.name, {**model_config.fixed_params, **hyperparameters}
            )
            model.fit(training_features, training_target)

            train_rmse = _rmse(training_target, model.predict(training_features))
            validation_rmse = _rmse(
                validation_target, model.predict(validation_features)
            )
            trial_history.append(
                {
                    "trial": trial.number,
                    "train_rmse": train_rmse,
                    "validation_rmse": validation_rmse,
                }
            )

            if (
                best_state["validation_rmse"] is None
                or validation_rmse < best_state["validation_rmse"]
            ):
                best_state["validation_rmse"] = validation_rmse
                best_state["model"] = model

            return validation_rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=model_config.n_trials)

        logging.info(
            f"{model_config.name}: best validation RMSE {study.best_value:.4f} "
            f"with params {study.best_params}"
        )
        return best_state["model"], trial_history
    except Exception as error:
        raise CustomException(str(error)) from error


def _rmse(true_values, predicted_values) -> float:
    """Compute root-mean-squared-error between true and predicted values."""
    return float(np.sqrt(mean_squared_error(true_values, predicted_values)))
