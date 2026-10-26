import numpy as np
import pandas as pd
import torch

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix, FEATURE_COLUMNS, TARGET_COLUMN
from demandpricing.models import (
    GradientBoostingForecaster, MLPForecaster, baseline_seasonal_naive_predict,
    compute_feature_scaling, dataframe_to_tensors,
)


def _featured_data():
    raw = generate_dataset(sku_ids=["A", "B"], n_days=200, seed=0)
    return build_feature_matrix(raw)


def test_baseline_seasonal_naive_matches_lag_7():
    df = _featured_data()
    predictions = baseline_seasonal_naive_predict(df)
    np.testing.assert_array_equal(predictions, df["lag_7"].to_numpy())


def test_gradient_boosting_forecaster_trains_and_predicts():
    df = _featured_data()
    model = GradientBoostingForecaster(n_estimators=20, max_depth=2).fit(df)
    predictions = model.predict(df)
    assert len(predictions) == len(df)
    assert np.all(np.isfinite(predictions))


def test_mlp_forward_pass_shape():
    model = MLPForecaster(n_features=len(FEATURE_COLUMNS))
    x = torch.randn(8, len(FEATURE_COLUMNS))
    out = model(x)
    assert out.shape == (8,)


def test_feature_scaling_produces_zero_mean_unit_std():
    df = _featured_data()
    mean, std = compute_feature_scaling(df)
    x, _ = dataframe_to_tensors(df, mean, std)
    np.testing.assert_allclose(x.numpy().mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(x.numpy().std(axis=0), 1.0, atol=1e-2)
