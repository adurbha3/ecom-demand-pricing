import pandas as pd

from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix
from demandpricing.tuning import random_search


def test_random_search_logs_one_trial_per_iteration():
    raw = generate_dataset(sku_ids=["A"], n_days=200, seed=0)
    featured = build_feature_matrix(raw)
    cutoff = featured["date"].max() - pd.Timedelta(days=30)
    train_df = featured[featured["date"] <= cutoff]
    val_df = featured[featured["date"] > cutoff]

    log = random_search(
        train_df, val_df,
        param_distributions={"n_estimators": [10, 20], "max_depth": [2, 3]},
        n_trials=4, seed=0,
    )
    assert len(log.records) == 4
    for record in log.records:
        assert record.metrics.rmse >= 0
