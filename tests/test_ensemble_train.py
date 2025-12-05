import numpy as np
from supply_chain_prediction.models import DelayPredictionModel
from supply_chain_prediction.models import EnsembleDelayPredictor


def make_tiny_data(n=50, d=6):
    rng = np.random.RandomState(42)
    X = rng.rand(n, d)
    y = rng.rand(n)
    return X, y


def test_ensemble_train_small_models():
    """Train the ensemble with tiny models to ensure `train` runs end-to-end."""
    X, y = make_tiny_data(n=40, d=8)

    # Split small validation data
    X_train, X_val = X[:30], X[30:35]
    y_train, y_val = y[:30], y[30:35]

    ensemble = EnsembleDelayPredictor(model_types=['xgboost', 'random_forest'])

    # Reduce complexity for fast tests
    if 'xgboost' in ensemble.models:
        ensemble.models['xgboost'].model.set_params(n_estimators=5)
    if 'random_forest' in ensemble.models:
        ensemble.models['random_forest'].model.set_params(n_estimators=10)

    metrics = ensemble.train(X_train, y_train, X_val, y_val)

    assert isinstance(metrics, dict)
    assert 'xgboost' in metrics or 'random_forest' in metrics
    # Ensure each model returned a metrics dict
    for v in metrics.values():
        assert isinstance(v, dict)
        assert 'train_rmse' in v
