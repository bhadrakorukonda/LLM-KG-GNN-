from .routes import router

__all__ = ["router", "app"]

def __getattr__(name):
    # Lazily fetch app to avoid circular import at import time.
    if name == "app":
        from backend.main import app as _app
        return _app
    raise AttributeError(name)
