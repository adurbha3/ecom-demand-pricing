#!/usr/bin/env python
"""Run a random hyperparameter search for the gradient-boosting forecaster
and print every trial's hyperparameters and validation metrics, plus the
best trial found -- the experiment log that makes a tuning run reproducible."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix
from demandpricing.tuning import random_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = generate_dataset(n_days=args.n_days, seed=args.seed)
    featured = build_feature_matrix(raw)
    cutoff = featured["date"].max() - pd.Timedelta(days=args.val_days)
    train_df = featured[featured["date"] <= cutoff]
    val_df = featured[featured["date"] > cutoff]

    print(f"Running {args.n_trials}-trial random search...\n")
    log = random_search(train_df, val_df, n_trials=args.n_trials, seed=args.seed)

    for record in log.records:
        print(f"trial {record.trial_id}: {record.hyperparams} -> RMSE={record.metrics.rmse:.2f} MAPE={record.metrics.mape:.1f}%")

    best = log.best(metric="rmse")
    print(f"\nBest trial: {best.trial_id} with hyperparameters {best.hyperparams}")
    print(f"  RMSE={best.metrics.rmse:.2f}  MAE={best.metrics.mae:.2f}  MAPE={best.metrics.mape:.1f}%")


if __name__ == "__main__":
    main()
