#!/usr/bin/env python
"""Estimate one SKU's price elasticity from its sales history and recommend
the revenue-maximizing price."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demandpricing.synthetic_data import generate_dataset
from demandpricing.pricing import estimate_elasticity, optimize_price


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku-id", type=str, default="SKU-001")
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = generate_dataset(n_days=args.n_days, seed=args.seed)
    sku_df = raw[raw["sku_id"] == args.sku_id]

    estimate = estimate_elasticity(raw, args.sku_id)
    print(f"Estimated elasticity for {args.sku_id}: {estimate.elasticity:.2f} (R^2={estimate.r_squared:.3f})")

    recent_non_promo = sku_df[sku_df["promo_flag"] == 0].tail(28)
    reference_price = float(recent_non_promo["price"].mean())
    reference_demand = float(recent_non_promo["units_sold"].mean())

    recommendation = optimize_price(
        reference_price=reference_price, reference_demand=reference_demand, elasticity=estimate.elasticity,
    )
    print(f"\nCurrent average price: ${reference_price:.2f}  (avg demand {reference_demand:.1f} units/day)")
    print(f"Recommended price:     ${recommendation.recommended_price:.2f}")
    print(f"Expected demand:       {recommendation.expected_demand:.1f} units/day")
    print(f"Expected revenue:      ${recommendation.expected_revenue:.2f}/day")

    current_revenue = reference_price * reference_demand
    uplift = (recommendation.expected_revenue - current_revenue) / current_revenue * 100
    print(f"Projected revenue uplift vs. current average price: {uplift:+.1f}%")


if __name__ == "__main__":
    main()
