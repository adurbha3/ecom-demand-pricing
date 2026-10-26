import numpy as np

from demandpricing.metrics import evaluate_forecast


def test_perfect_forecast_has_zero_error():
    actual = np.array([10, 20, 30])
    metrics = evaluate_forecast(actual, actual)
    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert metrics.mape == 0


def test_mae_and_rmse_known_values():
    actual = np.array([10, 20])
    predicted = np.array([12, 17])
    metrics = evaluate_forecast(actual, predicted)
    assert metrics.mae == 2.5
    assert abs(metrics.rmse - np.sqrt((4 + 9) / 2)) < 1e-9


def test_mape_floor_avoids_division_explosion():
    actual = np.array([0, 100])
    predicted = np.array([1, 100])
    metrics = evaluate_forecast(actual, predicted, mape_floor=1.0)
    assert metrics.mape < 100  # without the floor, the first point alone would dominate
