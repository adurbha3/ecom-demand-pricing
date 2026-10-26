"""Forecast accuracy metrics: MAE, RMSE, and MAPE (with a floor to avoid
dividing by near-zero actual sales)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> dict:
        return {"mae": self.mae, "rmse": self.rmse, "mape": self.mape}


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, mape_floor: float = 1.0) -> ForecastMetrics:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = actual - predicted

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = np.maximum(actual, mape_floor)
    mape = float(np.mean(np.abs(error) / denom) * 100)

    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape)
