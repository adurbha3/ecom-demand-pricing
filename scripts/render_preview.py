#!/usr/bin/env python
"""Render a PNG comparing actual sales against the gradient-boosting
forecast for one SKU over its validation window."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix, TARGET_COLUMN
from demandpricing.models import GradientBoostingForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku-id", type=str, default="SKU-001")
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="assets/forecast_preview.png")
    args = parser.parse_args()

    raw = generate_dataset(n_days=args.n_days, seed=args.seed)
    featured = build_feature_matrix(raw)
    cutoff = featured["date"].max() - pd.Timedelta(days=args.val_days)
    train_df = featured[featured["date"] <= cutoff]
    val_df = featured[featured["date"] > cutoff]

    model = GradientBoostingForecaster(n_estimators=150, max_depth=3, learning_rate=0.05).fit(train_df)

    sku_val = val_df[val_df["sku_id"] == args.sku_id]
    predictions = model.predict(sku_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sku_val["date"], sku_val[TARGET_COLUMN], label="Actual", color="#1f2937", linewidth=1.5)
    ax.plot(sku_val["date"], predictions, label="Forecast (gradient boosting)", color="#2563eb", linewidth=1.5, linestyle="--")
    ax.set_title(f"{args.sku_id}: actual vs. forecast daily units sold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved preview to {args.out}")


if __name__ == "__main__":
    main()
