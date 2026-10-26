import numpy as np
import pytest

from demandpricing.synthetic_data import generate_dataset
from demandpricing.pricing import estimate_elasticity, optimize_price


def test_estimate_elasticity_recovers_approximately_correct_sign_and_magnitude():
    raw = generate_dataset(sku_ids=["A"], n_days=500, elasticity=1.8, promo_probability=0.0, seed=0)
    estimate = estimate_elasticity(raw, "A")
    # The generator's elasticity is used as a negative exponent on price, so the
    # fitted log-log slope should be negative and roughly match its magnitude.
    assert estimate.elasticity < 0
    assert 1.0 < abs(estimate.elasticity) < 3.0


def test_estimate_elasticity_r_squared_is_plausible_with_seasonal_controls():
    """A first version of estimate_elasticity used only log(price) with no
    controls and returned R^2 under 0.02 on this dataset, because weekly and
    yearly seasonality swamp the price signal. With day-of-week/month fixed
    effects added as controls, R^2 should land in a plausible range."""
    raw = generate_dataset(sku_ids=["A"], n_days=500, elasticity=1.8, promo_probability=0.0, seed=0)
    estimate = estimate_elasticity(raw, "A")
    assert estimate.r_squared > 0.1


def test_estimate_elasticity_raises_with_too_little_data():
    raw = generate_dataset(sku_ids=["A"], n_days=5, seed=0)
    with pytest.raises(ValueError):
        estimate_elasticity(raw, "A")


def test_optimize_price_finds_price_above_zero_elasticity_midpoint():
    # With elastic demand (elasticity more negative than -1), lowering price from a
    # high reference price should increase revenue, so the optimizer should not
    # recommend the top of the range.
    recommendation = optimize_price(
        reference_price=25.0, reference_demand=100.0, elasticity=-1.8,
        price_range=(15.0, 35.0), n_candidates=200,
    )
    assert 15.0 <= recommendation.recommended_price <= 35.0
    assert recommendation.recommended_price < 35.0


def test_optimize_price_with_unit_cost_maximizes_profit_not_revenue():
    rec_revenue = optimize_price(reference_price=25.0, reference_demand=100.0, elasticity=-1.8)
    rec_profit = optimize_price(reference_price=25.0, reference_demand=100.0, elasticity=-1.8, unit_cost=10.0)
    assert rec_profit.expected_profit is not None
    # Optimizing for profit above a nonzero cost should recommend a price at or
    # above the pure-revenue optimum, since marginal units below cost hurt profit.
    assert rec_profit.recommended_price >= rec_revenue.recommended_price - 1e-6
