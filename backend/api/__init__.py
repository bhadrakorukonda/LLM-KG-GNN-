from __future__ import annotations
import os, json, io
from pathlib import Path
from typing import Dict, List, Tuple
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Import retriever (our shim provides load_graph, G, NODE_TEXT after /reload)
from backend import retriever as R  # type: ignore

# Compute repo root and data dir without relying on R.DATA
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]  # .../repo/backend/api/__init__.py -> repo
_DATA_DIR  = Path(os.environ.get("GRAIL_DATA", _REPO_ROOT / "data"))

_EDGES = _DATA_DIR / "kg_edges.tsv"
_TEXTS = _DATA_DIR / "node_texts.jsonl"

app = FastAPI(title="Graph RAG Sprint API")

def _count_edges_nodes_from_tsv(path: Path) -> Tuple[int, int]:
    """Len edges, len unique nodes from tab-separated triples."""
    if not path.exists():
        return 0, 0
    edges = 0
    nodes = set()
    with io.open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                edges += 1
                nodes.add(h); nodes.add(t)
    return edges, len(nodes)

def _count_texts_from_jsonl(path: Path) -> int:
    if not path.exists(): 
        return 0
    n = 0
    with io.open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

class AskBody(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 1

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reload")
def reload_graph():
    # Ensure data dir exists
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Load via retriever shim (fills R.G and R.NODE_TEXT)
    try:
        R.load_graph(str(_EDGES), str(_TEXTS))  # type: ignore[attr-defined]
    except Exception as e:
        # Still return file-based counts so UI isn’t blocked
        e_str = f"{type(e).__name__}: {e}"
        file_edges, file_nodes = _count_edges_nodes_from_tsv(_EDGES)
        texts = _count_texts_from_jsonl(_TEXTS)
        return {"nodes": file_nodes, "edges": file_edges, "texts": texts, "note": "file-counts (R.load_graph failed)", "error": e_str}

    # Prefer retriever counters; fall back to file-based counts
    nodes = texts = edges = None
    try:
        nodes = len(getattr(R, "G").nodes)     # type: ignore
        edges = len(getattr(R, "G").edges)     # type: ignore
    except Exception:
        file_edges, file_nodes = _count_edges_nodes_from_tsv(_EDGES)
        nodes, edges = file_nodes, file_edges

    try:
        texts = len(getattr(R, "NODE_TEXT"))   # type: ignore
    except Exception:
        texts = _count_texts_from_jsonl(_TEXTS)

    return {"nodes": nodes or 0, "edges": edges or 0, "texts": texts or 0}

# Keep a minimal /ask so api_linked can import this module cleanly.
# Your prioritized /ask lives in app/backend/api_linked.py and will be hit first.
@app.post("/ask")
def ask(body: AskBody):
    # Non-priority fallback; your api_linked /ask should shadow this.
    return {
        "answer": "DEFAULT_ANSWER",
        "ctx": {
            "seeds": [],
            "paths": [],
            "contexts": [],
            "local_facts": [],
            "node_notes": []
        },
        "source": "base"
    }