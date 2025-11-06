# backend/api.py
from __future__ import annotations
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel

from .retriever import load_graph, build_cache, detect_seeds_from_question, shortest_paths_multi_seed
from . import llm

app = FastAPI(title="Graph LLM–KG API", version="0.1.0")

_state: Dict[str, Any] = {
    "caches": None,
    "cache_mode": False,
    "model_name": os.environ.get("OLLAMA_MODEL", "llama3")
}

def _get_caches():
    if _state["caches"] is None:
        _state["caches"] = load_graph()
        _state["cache_mode"] = True
    return _state["caches"]

class AskRequest(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 2
    allowed_relations: Optional[List[str]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"name": app.title, "version": app.version}

@app.get("/__fingerprint")
def fingerprint():
    _get_caches()
    return {"cache_mode": bool(_state["cache_mode"])}

@app.get("/stats")
def stats():
    c = _get_caches()
    nnz = int(c.adj.nnz)
    return {
        "nodes": int(c.adj.shape[0]),
        "edges_directed": nnz,
        "edges_undirected_estimate": nnz // 2
    }

@app.post("/reload")
def reload_graph():
    edges = os.environ.get("KG_EDGES", "").strip()
    texts = os.environ.get("KG_NODE_TEXTS", "").strip() or None
    if not edges:
        return {"status": "error", "note": "KG_EDGES not set"}
    _state["caches"] = build_cache(edges, texts)
    _state["cache_mode"] = True
    return {"status": "ok", "edges": edges, "texts": texts or None, "cache": True}

@app.get("/models")
def models():
    try:
        ms = llm.list_models()
        return {"status": "ok", "models": ms}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/ask")
def ask(req: AskRequest, dry_run: bool = Query(False)):
    caches = _get_caches()
    seeds = detect_seeds_from_question(req.question, caches.nid2i.keys())
    paths = shortest_paths_multi_seed(
        caches,
        seed_ids=seeds,
        topk_paths=req.topk_paths,
        max_hops=req.max_hops,
        neighbor_expand=req.neighbor_expand,
        allowed_relation_names=req.allowed_relations
    )

    ctx = {
        "seeds": seeds,
        "paths": paths,
        "contexts": [],
        "local_facts": [],
        "node_notes": [
            f"{nid}: {caches.text_by_i[caches.nid2i.get(nid, 0)]}"
            for nid in seeds if nid in caches.nid2i
        ],
    }

    if dry_run:
        return {"answer": "", "source": "dry_run", "ctx": ctx, "note": "dry_run (LLM skipped)"}

    try:
        model = os.environ.get("OLLAMA_MODEL", "llama3")
        answer = llm.generate_answer(model, req.question, paths)
        return {"answer": answer, "source": "llm", "ctx": ctx}
    except Exception as e:
        return {"answer": "", "source": "llm_error", "error": str(e), "ctx": ctx}

