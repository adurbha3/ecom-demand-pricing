"""A structured experiment log: one record per training run, with its
hyperparameters and resulting metrics, so a forecasting experiment is
reproducible and comparable to every other run rather than living only in
a terminal scrollback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from demandpricing.metrics import ForecastMetrics


@dataclass
class ExperimentRecord:
    trial_id: int
    model_name: str
    hyperparams: dict
    metrics: ForecastMetrics
    timestamp: float = field(default_factory=time.time)

    def as_row(self) -> dict:
        row = {"trial_id": self.trial_id, "model_name": self.model_name, "timestamp": self.timestamp}
        row.update({f"param_{k}": v for k, v in self.hyperparams.items()})
        row.update(self.metrics.as_dict())
        return row


class ExperimentLog:
    def __init__(self):
        self.records: list[ExperimentRecord] = []

    def log(self, model_name: str, hyperparams: dict, metrics: ForecastMetrics) -> ExperimentRecord:
        record = ExperimentRecord(
            trial_id=len(self.records), model_name=model_name, hyperparams=hyperparams, metrics=metrics
        )
        self.records.append(record)
        return record

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row() for r in self.records])

    def best(self, metric: str = "rmse", minimize: bool = True) -> ExperimentRecord | None:
        if not self.records:
            return None
        key = (lambda r: getattr(r.metrics, metric)) if minimize else (lambda r: -getattr(r.metrics, metric))
        return min(self.records, key=key)
