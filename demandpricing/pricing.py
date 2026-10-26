"""Price elasticity estimation and a revenue/profit-maximizing price
optimizer built on top of it -- this is the "pricing" half of a demand
forecasting system, turning a forecast into an actionable pricing decision
rather than stopping at the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ElasticityEstimate:
    sku_id: str
    elasticity: float  # the log-log regression slope; negative for a normal good
    intercept: float
    r_squared: float


def estimate_elasticity(df: pd.DataFrame, sku_id: str) -> ElasticityEstimate:
    """Fit log(units_sold) = intercept + elasticity * log(price) + day-of-week
    and month fixed effects, for one SKU's non-promotional days (promo days
    are excluded so a temporary discount's demand spike does not get
    attributed to the regular price/demand relationship).

    A first version of this function regressed log(units_sold) on log(price)
    alone and returned an R^2 near 0.01 on this project's synthetic data --
    technically a fitted line, but one explaining almost none of the
    variance, because weekly and yearly seasonality (built into the
    generator) swamps the price signal when nothing else is controlled for.
    Adding day-of-week and month fixed effects as controls raised R^2 into
    a plausible range (commonly 0.5+ on this dataset) and is why they are
    included below rather than only the bivariate fit.
    """
    sku_df = df[(df["sku_id"] == sku_id) & (df["promo_flag"] == 0) & (df["units_sold"] > 0)].copy()
    if len(sku_df) < 10:
        raise ValueError(f"not enough non-promotional data points for {sku_id} to estimate elasticity")

    log_price = np.log(sku_df["price"].to_numpy()).reshape(-1, 1)
    log_units = np.log(sku_df["units_sold"].to_numpy())

    day_of_week = sku_df["date"].dt.day_of_week.to_numpy().reshape(-1, 1)
    month = sku_df["date"].dt.month.to_numpy().reshape(-1, 1)
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    controls = encoder.fit_transform(np.hstack([day_of_week, month]))

    features = np.hstack([log_price, controls])
    model = LinearRegression().fit(features, log_units)

    r_squared = model.score(features, log_units)
    elasticity = float(model.coef_[0])  # the log(price) column is first

    return ElasticityEstimate(
        sku_id=sku_id, elasticity=elasticity, intercept=float(model.intercept_), r_squared=float(r_squared)
    )


@dataclass
class PriceRecommendation:
    recommended_price: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float | None
    price_grid: np.ndarray
    revenue_grid: np.ndarray


def optimize_price(
    reference_price: float,
    reference_demand: float,
    elasticity: float,
    price_range: tuple[float, float] | None = None,
    n_candidates: int = 200,
    unit_cost: float | None = None,
) -> PriceRecommendation:
    """Sweep a grid of candidate prices, project demand at each via the
    estimated elasticity, and pick the price maximizing revenue (or profit,
    if unit_cost is given)."""
    low, high = price_range or (reference_price * 0.6, reference_price * 1.4)
    price_grid = np.linspace(low, high, n_candidates)

    demand_grid = reference_demand * (price_grid / reference_price) ** elasticity
    revenue_grid = price_grid * demand_grid

    if unit_cost is not None:
        objective_grid = (price_grid - unit_cost) * demand_grid
    else:
        objective_grid = revenue_grid

    best_index = int(np.argmax(objective_grid))
    best_price = float(price_grid[best_index])
    best_demand = float(demand_grid[best_index])
    best_revenue = float(revenue_grid[best_index])
    best_profit = float(objective_grid[best_index]) if unit_cost is not None else None

    return PriceRecommendation(
        recommended_price=best_price, expected_demand=best_demand, expected_revenue=best_revenue,
        expected_profit=best_profit, price_grid=price_grid, revenue_grid=revenue_grid,
    )
