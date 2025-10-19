from fastapi.testclient import TestClient
from backend.main import app


def test_ask_dry_run_returns_ctx_only():
    client = TestClient(app)
    r = client.post("/ask?dry_run=true", json={
        "question": "Who co-authored with Carol?",
        "topk_paths": 1,
        "max_hops": 1,
        "neighbor_expand": 0,
        "use_rerank": False,
        "model": "ollama/llama3",
    })
    assert r.status_code == 200
    j = r.json()
    assert "ctx" in j
    assert "answer" not in j


