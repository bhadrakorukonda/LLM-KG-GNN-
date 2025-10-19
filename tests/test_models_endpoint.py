import httpx
from fastapi.testclient import TestClient
from backend.main import app


def test_models_endpoint(monkeypatch):
    class DummyResp:
        status_code = 200
        def json(self):
            return {"models": [{"name": "llama3"}]}

    def fake_get(self, url):
        return DummyResp()

    # Patch httpx.Client.get
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setattr(httpx.Client, "get", fake_get)

    client = TestClient(app)
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == ["llama3"]


