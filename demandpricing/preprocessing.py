"""Data cleaning and feature engineering for the demand forecasting models:
calendar features, lagged/rolling demand history (computed per SKU, shifted
to avoid leaking the current day's own sales into its own features), and
outlier capping on the raw sales column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14, 28]


def cap_outliers(df: pd.DataFrame, column: str = "units_sold", n_std: float = 4.0) -> pd.DataFrame:
    """Cap values beyond n_std standard deviations (computed per SKU) rather
    than dropping them, since an extreme sales day is still a real data
    point -- just one that should not be allowed to dominate training."""
    df = df.copy()

    def _cap(group: pd.Series) -> pd.Series:
        mean, std = group.mean(), group.std()
        if std == 0 or np.isnan(std):
            return group
        lower, upper = mean - n_std * std, mean + n_std * std
        return group.clip(lower=max(lower, 0), upper=upper)

    df[column] = df.groupby("sku_id")[column].transform(_cap)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_year"] = df["date"].dt.day_of_year
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, lags: list[int] = LAG_DAYS, windows: list[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "date"]).copy()
    grouped = df.groupby("sku_id")["units_sold"]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Rolling stats computed on the already-shifted lag_1 series, so a
    # window never includes the current day's own sales.
    shifted = grouped.shift(1)
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df["sku_id"]).transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df["sku_id"]).transform(lambda s: s.rolling(window, min_periods=2).std())
        )

    return df


FEATURE_COLUMNS = (
    ["price", "promo_flag", "holiday_flag", "day_of_week", "day_of_year", "month", "is_weekend"]
    + [f"lag_{lag}" for lag in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
)
TARGET_COLUMN = "units_sold"


def build_feature_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: cap outliers, add calendar/lag/rolling features, and
    drop the per-SKU startup rows that do not yet have full lag history."""
    df = cap_outliers(raw_df)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = df.dropna(subset=[f"lag_{max(LAG_DAYS)}", f"rolling_std_{max(ROLLING_WINDOWS)}"])
    return df.reset_index(drop=True)
