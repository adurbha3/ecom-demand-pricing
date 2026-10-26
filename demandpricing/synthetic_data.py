"""A seeded synthetic daily ecommerce sales generator: multiple SKUs with
trend, weekly and yearly seasonality, price elasticity, promotions, holiday
spikes, and noise -- a stand-in for the kind of marketplace sales data a
demand forecasting and pricing model would be trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAY_DAYS_OF_YEAR = {1, 47, 186, 359, 360, 361, 328}  # New Year's, pre-Valentine's, July 4th, Black Friday week, Cyber Monday
REFERENCE_PRICE = 25.0
BASE_DEMAND = 120.0
ELASTICITY = 1.8  # % demand drop per % price increase, > 1 is "elastic"


def _seasonal_multiplier(day_of_week: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_week - 5) / 7)  # weekend lift
    yearly = 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_year - 300) / 365)  # winter lift
    return weekly * yearly


def generate_sku_series(
    sku_id: str,
    n_days: int = 730,
    start_date: str = "2025-01-01",
    base_demand: float = BASE_DEMAND,
    elasticity: float = ELASTICITY,
    reference_price: float = REFERENCE_PRICE,
    trend_per_day: float = 0.03,
    promo_probability: float = 0.06,
    promo_discount: float = 0.20,
    noise_sigma: float = 8.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_of_week = dates.day_of_week.to_numpy()
    day_of_year = dates.day_of_year.to_numpy()

    # Price: a slow random walk around the reference price, with occasional promo discounts.
    price_walk = reference_price + np.cumsum(rng.normal(0, 0.05, size=n_days))
    price_walk = np.clip(price_walk, reference_price * 0.6, reference_price * 1.4)
    promo_flag = rng.random(n_days) < promo_probability
    price = np.where(promo_flag, price_walk * (1 - promo_discount), price_walk)

    holiday_flag = np.isin(day_of_year, list(HOLIDAY_DAYS_OF_YEAR))

    trend = 1.0 + trend_per_day * (np.arange(n_days) / 365.0)
    seasonal = _seasonal_multiplier(day_of_week, day_of_year)
    price_effect = (price / reference_price) ** (-elasticity)
    holiday_effect = np.where(holiday_flag, 1.8, 1.0)
    promo_effect = np.where(promo_flag, 1.35, 1.0)

    expected_demand = base_demand * trend * seasonal * price_effect * holiday_effect * promo_effect
    units_sold = np.clip(expected_demand + rng.normal(0, noise_sigma, size=n_days), 0, None).round()

    return pd.DataFrame({
        "date": dates,
        "sku_id": sku_id,
        "price": np.round(price, 2),
        "promo_flag": promo_flag.astype(int),
        "holiday_flag": holiday_flag.astype(int),
        "units_sold": units_sold.astype(int),
    })


def generate_dataset(
    sku_ids: list[str] | None = None, n_days: int = 730, seed: int | None = None, **kwargs
) -> pd.DataFrame:
    """Generate a multi-SKU dataset. By default each SKU gets its own randomized
    base_demand and elasticity (for realistic cross-SKU variety); pass either
    explicitly via kwargs to fix it for every SKU instead (e.g. in a test that
    needs a known elasticity to check against)."""
    sku_ids = sku_ids or [f"SKU-{i:03d}" for i in range(1, 6)]
    rng = np.random.default_rng(seed)
    fixed_base_demand = kwargs.pop("base_demand", None)
    fixed_elasticity = kwargs.pop("elasticity", None)

    frames = []
    for sku_id in sku_ids:
        sub_seed = int(rng.integers(0, 2**31 - 1))
        base_demand = fixed_base_demand if fixed_base_demand is not None else BASE_DEMAND * rng.uniform(0.5, 2.0)
        elasticity = fixed_elasticity if fixed_elasticity is not None else rng.uniform(1.2, 2.5)
        frames.append(
            generate_sku_series(
                sku_id, n_days=n_days, base_demand=base_demand, elasticity=elasticity,
                seed=sub_seed, **kwargs,
            )
        )
    return pd.concat(frames, ignore_index=True)
