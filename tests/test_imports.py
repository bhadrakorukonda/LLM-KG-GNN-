def test_backend_api_exports_app():
    import importlib
    mod = importlib.import_module("backend.api")
    assert hasattr(mod, "app"), "backend.api must export 'app'"


