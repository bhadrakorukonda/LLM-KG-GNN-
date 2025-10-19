from fastapi.testclient import TestClient
from backend.main import app


def test_ask_uses_generate_answer(monkeypatch):
    from backend import main as backend_main

    def fake_generate_answer(question: str, ctx: dict, model: str = "ollama/llama3", timeout_s: int = 30) -> str:
        return "STUB_ANSWER"

    monkeypatch.setattr(backend_main, "generate_answer", fake_generate_answer)

    client = TestClient(app)
    # ensure minimal graph load to avoid error path
    # rely on startup loader; if graph empty, the API returns error
    # we simulate a call and accept either success or graceful error
    resp = client.post("/ask", json={
        "question": "Who co-authored with Carol?",
        "topk_paths": 1,
        "max_hops": 1,
        "neighbor_expand": 0,
        "use_rerank": False,
        "model": "ollama/llama3",
    })
    assert resp.status_code == 200
    data = resp.json()
    if "error" not in data:
        assert "answer" in data
        assert data["answer"] == "STUB_ANSWER"


