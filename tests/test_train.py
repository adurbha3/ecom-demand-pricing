from demandpricing.synthetic_data import generate_dataset
from demandpricing.preprocessing import build_feature_matrix
from demandpricing.train import MLPTrainConfig, train_mlp_forecaster, predict_mlp


def test_train_mlp_runs_and_predicts():
    raw = generate_dataset(sku_ids=["A"], n_days=150, seed=0)
    df = build_feature_matrix(raw)
    model, mean, std = train_mlp_forecaster(df, MLPTrainConfig(epochs=2, batch_size=16))
    predictions = predict_mlp(model, df, mean, std)
    assert len(predictions) == len(df)
    assert all(p == p for p in predictions)  # no NaNs
