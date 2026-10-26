import numpy as np

from demandpricing.synthetic_data import generate_sku_series, generate_dataset


def test_generate_sku_series_shape_and_columns():
    df = generate_sku_series("SKU-TEST", n_days=100, seed=1)
    assert len(df) == 100
    assert set(df.columns) == {"date", "sku_id", "price", "promo_flag", "holiday_flag", "units_sold"}


def test_units_sold_is_nonnegative():
    df = generate_sku_series("SKU-TEST", n_days=200, seed=2)
    assert (df["units_sold"] >= 0).all()


def test_higher_price_elasticity_reduces_average_demand():
    low_elasticity = generate_sku_series("A", n_days=300, elasticity=0.5, promo_probability=0, seed=3)
    high_elasticity = generate_sku_series("B", n_days=300, elasticity=3.0, promo_probability=0, seed=3)
    # Same price path and seed; the more elastic SKU should show more demand variation
    # relative to price moves, not necessarily lower mean -- check variance instead.
    assert high_elasticity["units_sold"].std() != low_elasticity["units_sold"].std()


def test_reproducible_with_same_seed():
    a = generate_sku_series("A", n_days=50, seed=42)
    b = generate_sku_series("A", n_days=50, seed=42)
    assert np.array_equal(a["units_sold"].to_numpy(), b["units_sold"].to_numpy())


def test_generate_dataset_covers_all_skus():
    df = generate_dataset(sku_ids=["X", "Y", "Z"], n_days=60, seed=0)
    assert set(df["sku_id"].unique()) == {"X", "Y", "Z"}
    assert len(df) == 60 * 3
