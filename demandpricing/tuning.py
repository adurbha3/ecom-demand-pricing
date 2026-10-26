"""Random hyperparameter search for the gradient-boosting forecaster, with
every trial's hyperparameters and validation metrics recorded in an
ExperimentLog."""
from __future__ import annotations

import numpy as np
import pandas as pd

from demandpricing.experiment_log import ExperimentLog
from demandpricing.metrics import evaluate_forecast
from demandpricing.models import GradientBoostingForecaster
from demandpricing.preprocessing import TARGET_COLUMN

DEFAULT_PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}


def random_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    param_distributions: dict = DEFAULT_PARAM_DISTRIBUTIONS,
    n_trials: int = 8,
    seed: int | None = None,
) -> ExperimentLog:
    rng = np.random.default_rng(seed)
    log = ExperimentLog()

    for _ in range(n_trials):
        hyperparams = {name: rng.choice(values).item() for name, values in param_distributions.items()}
        model = GradientBoostingForecaster(**hyperparams).fit(train_df)
        predictions = model.predict(val_df)
        metrics = evaluate_forecast(val_df[TARGET_COLUMN].to_numpy(), predictions)
        log.log(model_name="gradient_boosting", hyperparams=hyperparams, metrics=metrics)

    return log
