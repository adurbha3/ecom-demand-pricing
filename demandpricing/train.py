"""Training loop for the MLP forecaster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from demandpricing.models import MLPForecaster, compute_feature_scaling, dataframe_to_tensors


@dataclass
class MLPTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 32


def train_mlp_forecaster(
    train_df: pd.DataFrame, config: MLPTrainConfig = MLPTrainConfig()
) -> tuple[MLPForecaster, np.ndarray, np.ndarray]:
    feature_mean, feature_std = compute_feature_scaling(train_df)
    x, y = dataframe_to_tensors(train_df, feature_mean, feature_std)

    model = MLPForecaster(hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)

    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model, feature_mean, feature_std


def predict_mlp(model: MLPForecaster, df: pd.DataFrame, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    model.eval()
    x, _ = dataframe_to_tensors(df, feature_mean, feature_std)
    with torch.no_grad():
        return model(x).numpy()
