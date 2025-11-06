# tests/test_ctx_smoke.py
from backend.retriever import load_graph, G, NODE_TEXT
import requests

def test_graph_counts():
    load_graph()
    assert len(G.edges) > 0
    assert len(NODE_TEXT) > 0

def test_ask_dry_run():
    r = requests.post("http://localhost:8001/ask?dry_run=true", json={
        "question":"Who co-authored a paper with Carol?",
        "topk_paths":5, "max_hops":3, "neighbor_expand":2
    })
    r.raise_for_status()
    ctx = r.json()["ctx"]
    assert ctx["paths"], "expected non-empty paths"
