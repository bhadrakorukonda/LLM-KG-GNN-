import sys
from pathlib import Path

# Add repo root to sys.path (works locally and in CI)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient as _BaseTestClient
from backend.main import app

# Patch TestClient.get to bypass httpx.Client.get (which tests monkeypatch)
# and call .request() directly, which is more robust to stubs without **kwargs.
def _safe_get(self, url, **kwargs):
    return _BaseTestClient.request(self, "GET", url, **kwargs)

# Apply once at import
try:
    from starlette.testclient import TestClient as _StarletteTestClient
    setattr(_StarletteTestClient, "get", _safe_get)
except Exception:
    pass

@pytest.fixture(scope="session")
def client():
    return _BaseTestClient(app)
