from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_ask_schema_and_types():
    r = client.post("/ask", json={"question": "Who co-authored a paper with Carol?"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["answer"], str)
    assert isinstance(data["paths"], list)
    assert isinstance(data["context"], str)
    assert isinstance(data["model"], str)
    if data["paths"]:
        assert all(isinstance(p, list) for p in data["paths"])
        assert all(all(isinstance(x, str) for x in p) for p in data["paths"])
