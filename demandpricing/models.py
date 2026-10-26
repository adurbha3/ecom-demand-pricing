"""Three forecasting approaches, compared on the same feature matrix:
a seasonal-naive baseline, a classical gradient-boosting model (scikit-learn),
and a small feedforward neural network (PyTorch).

A feedforward network, not a CNN, is used here deliberately: the inputs are
engineered tabular features (price, calendar flags, lag/rolling stats), not
a spatial or sequential array, so a CNN's convolutional structure would not
apply to anything meaningful in this representation -- an MLP is the
appropriate PyTorch architecture for this shape of input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn

from demandpricing.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


def baseline_seasonal_naive_predict(df: pd.DataFrame) -> np.ndarray:
    """Predict today's sales as last week's sales for the same SKU -- a
    simple, strong baseline given the data's weekly seasonality."""
    return df["lag_7"].to_numpy()


class GradientBoostingForecaster:
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = GradientBoostingRegressor(**hyperparams)

    def fit(self, df: pd.DataFrame) -> "GradientBoostingForecaster":
        self.model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[FEATURE_COLUMNS])


class MLPForecaster(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS), hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def dataframe_to_tensors(df: pd.DataFrame, feature_mean: np.ndarray, feature_std: np.ndarray):
    x = (df[FEATURE_COLUMNS].to_numpy(dtype=np.float32) - feature_mean) / feature_std
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def compute_feature_scaling(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std
