def test_import_supply_chain_prediction():
    """Ensure the `supply_chain_prediction` package imports without errors."""
    import importlib

    m = importlib.import_module('supply_chain_prediction')
    # Basic sanity checks
    assert m is not None
    assert hasattr(m, '__file__')
