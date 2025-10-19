from fastapi.testclient import TestClient
from backend.main import app
from backend.services.graph import graph_store


def test_smoke_ask_returns_paths_in_dry_run(tmp_path):
    # tiny graph
    edges = tmp_path / "edges.tsv"
    edges.write_text("Alice\tcoauthored_with\tCarol\n", encoding="utf-8")
    nodes = tmp_path / "node_texts.jsonl"
    nodes.write_text('{"id":"Alice","text":"Author"}\n', encoding="utf-8")

    # load
    stats = graph_store.reload(str(edges), str(nodes))
    assert stats["nodes"] >= 2 and stats["edges"] >= 1

    client = TestClient(app)
    r = client.post("/ask?dry_run=true", json={
        "question": "Who co-authored with Carol?",
        "topk_paths": 3,
        "max_hops": 2,
        "neighbor_expand": 1,
        "use_rerank": False,
        "model": "ollama/llama3",
    })
    assert r.status_code == 200
    j = r.json()
    assert "ctx" in j
    assert j["ctx"].get("paths") is not None
    assert isinstance(j["ctx"]["paths"], list)
    assert len(j["ctx"]["paths"]) >= 1


