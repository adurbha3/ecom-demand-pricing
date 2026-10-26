#!/usr/bin/env python
"""Generate synthetic sales data, build features, and compare a seasonal-naive
baseline, a gradient-boosting model, and a small PyTorch MLP on held-out
(time-based) validation data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix, TARGET_COLUMN
from demandpricing.models import GradientBoostingForecaster, baseline_seasonal_naive_predict
from demandpricing.train import MLPTrainConfig, train_mlp_forecaster, predict_mlp
from demandpricing.metrics import evaluate_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--n-skus", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Generating {args.n_days} days of sales data for {args.n_skus} SKUs...")
    sku_ids = [f"SKU-{i:03d}" for i in range(1, args.n_skus + 1)]
    raw = generate_dataset(sku_ids=sku_ids, n_days=args.n_days, seed=args.seed)
    featured = build_feature_matrix(raw)

    cutoff = featured["date"].max() - pd.Timedelta(days=args.val_days)
    train_df = featured[featured["date"] <= cutoff]
    val_df = featured[featured["date"] > cutoff]
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows (last {args.val_days} days per SKU)\n")

    actual = val_df[TARGET_COLUMN].to_numpy()

    baseline_pred = baseline_seasonal_naive_predict(val_df)
    baseline_metrics = evaluate_forecast(actual, baseline_pred)
    print(f"Seasonal-naive baseline:  MAE={baseline_metrics.mae:.2f}  RMSE={baseline_metrics.rmse:.2f}  MAPE={baseline_metrics.mape:.1f}%")

    gbm = GradientBoostingForecaster(n_estimators=150, max_depth=3, learning_rate=0.05).fit(train_df)
    gbm_pred = gbm.predict(val_df)
    gbm_metrics = evaluate_forecast(actual, gbm_pred)
    print(f"Gradient boosting:        MAE={gbm_metrics.mae:.2f}  RMSE={gbm_metrics.rmse:.2f}  MAPE={gbm_metrics.mape:.1f}%")

    mlp, feature_mean, feature_std = train_mlp_forecaster(train_df, MLPTrainConfig(epochs=args.epochs))
    mlp_pred = predict_mlp(mlp, val_df, feature_mean, feature_std)
    mlp_metrics = evaluate_forecast(actual, mlp_pred)
    print(f"PyTorch MLP:              MAE={mlp_metrics.mae:.2f}  RMSE={mlp_metrics.rmse:.2f}  MAPE={mlp_metrics.mape:.1f}%")


if __name__ == "__main__":
    main()
