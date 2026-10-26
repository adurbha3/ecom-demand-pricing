from demandpricing.experiment_log import ExperimentLog
from demandpricing.metrics import ForecastMetrics


def test_log_records_trials_in_order():
    log = ExperimentLog()
    log.log("model_a", {"lr": 0.1}, ForecastMetrics(mae=5, rmse=6, mape=10))
    log.log("model_a", {"lr": 0.2}, ForecastMetrics(mae=3, rmse=4, mape=8))
    assert len(log.records) == 2
    assert log.records[0].trial_id == 0
    assert log.records[1].trial_id == 1


def test_best_picks_lowest_rmse_by_default():
    log = ExperimentLog()
    log.log("a", {}, ForecastMetrics(mae=5, rmse=6, mape=10))
    log.log("b", {}, ForecastMetrics(mae=3, rmse=4, mape=8))
    log.log("c", {}, ForecastMetrics(mae=9, rmse=9, mape=20))
    assert log.best(metric="rmse").model_name == "b"


def test_best_returns_none_when_empty():
    assert ExperimentLog().best() is None


def test_to_dataframe_has_one_row_per_trial_with_hyperparams_and_metrics():
    log = ExperimentLog()
    log.log("a", {"max_depth": 3}, ForecastMetrics(mae=5, rmse=6, mape=10))
    df = log.to_dataframe()
    assert len(df) == 1
    assert "param_max_depth" in df.columns
    assert "rmse" in df.columns
