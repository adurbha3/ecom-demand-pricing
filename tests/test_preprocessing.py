import numpy as np
import pandas as pd

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import (
    cap_outliers, add_calendar_features, add_lag_and_rolling_features, build_feature_matrix, FEATURE_COLUMNS,
)


def test_cap_outliers_limits_extreme_values_per_sku():
    df = pd.DataFrame({
        "sku_id": ["A"] * 10,
        "units_sold": [10, 11, 9, 10, 12, 11, 10, 9, 10, 1000],
    })
    capped = cap_outliers(df, n_std=2.0)
    assert capped["units_sold"].max() < 1000


def test_add_calendar_features_adds_expected_columns():
    df = generate_dataset(n_days=30, seed=0)
    df = add_calendar_features(df)
    assert {"day_of_week", "day_of_year", "month", "is_weekend"} <= set(df.columns)
    assert df["is_weekend"].isin([0, 1]).all()


def test_lag_features_shift_correctly_within_sku():
    raw = generate_dataset(sku_ids=["A"], n_days=20, seed=0)
    df = add_lag_and_rolling_features(raw, lags=[1], windows=[])
    # lag_1 on day i should equal units_sold on day i-1
    shifted = df["units_sold"].shift(1).to_numpy()
    np.testing.assert_array_equal(df["lag_1"].to_numpy()[1:], shifted[1:])
    assert np.isnan(df["lag_1"].iloc[0])


def test_lag_features_do_not_cross_sku_boundaries():
    raw = generate_dataset(sku_ids=["A", "B"], n_days=10, seed=0)
    df = add_lag_and_rolling_features(raw, lags=[1], windows=[])
    first_row_of_b = df[df["sku_id"] == "B"].iloc[0]
    assert np.isnan(first_row_of_b["lag_1"])  # must not pick up SKU A's last value


def test_build_feature_matrix_drops_startup_rows_and_has_no_nans():
    raw = generate_dataset(sku_ids=["A"], n_days=100, seed=0)
    featured = build_feature_matrix(raw)
    assert len(featured) < 100  # startup rows (no full lag/rolling history) are dropped
    assert not featured[FEATURE_COLUMNS].isna().any().any()
