# backend/server.py
from __future__ import annotations
import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

# retriever bits
from .retriever import (
    load_graph,
    build_cache,
    detect_seeds_from_question,
    shortest_paths_multi_seed,
    has_cache,  # new: quick check so we don't 500 on empty cache
)

# LLM wiring (Ollama)
from . import llm


app = FastAPI(title="Graph LLM–KG API", version="0.1.0")

_state: Dict[str, Any] = {
    "caches": None,
    "cache_mode": False,
    "model_name": os.environ.get("OLLAMA_MODEL", "llama3"),
}


def _get_caches():
    """
    Lazy-load caches if present; otherwise stay None so endpoints can
    respond gracefully without throwing 500s.
    """
    if _state["caches"] is None:
        if has_cache():
            _state["caches"] = load_graph()  # tolerant loader per retriever.py
            _state["cache_mode"] = True
        else:
            _state["cache_mode"] = False
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


@app.get("/config")
def config():
    """
    Quick visibility into what the *server process* sees.
    Super useful when envs differ across terminals.
    """
    return {
        "KG_EDGES": os.environ.get("KG_EDGES"),
        "KG_NODE_TEXTS": os.environ.get("KG_NODE_TEXTS"),
        "OLLAMA_HOST": os.environ.get("OLLAMA_HOST"),
        "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3"),
        "cache_mode": bool(_state["cache_mode"]),
        "has_cache_files": has_cache(),
    }


@app.get("/__fingerprint")
def fingerprint():
    _ = _get_caches()  # ensures cache_mode reflects reality
    return {"cache_mode": bool(_state["cache_mode"])}


@app.get("/stats")
def stats():
    c = _get_caches()
    if c is None:
        return {"status": "error", "note": "no cache yet; set KG_EDGES and POST /reload"}
    return {
        "nodes": int(c.adj.shape[0]),
        "edges_directed": int(c.adj.nnz),
        "edges_undirected_estimate": int(c.adj.nnz) // 2,
    }


@app.post("/reload")
def reload_graph():
    """
    Rebuild cache from env-provided KG paths.
    Safe to call repeatedly when swapping datasets.
    """
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
        return {"status": "ok", "models": llm.list_models()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/ask")
def ask(req: AskRequest, dry_run: bool = Query(False)):
    """
    Multi-seed BFS + (optional) relation filter -> LLM answer via Ollama.
    With dry_run=true, returns only the retrieved paths & ctx.
    """
    caches = _get_caches()
    if caches is None:
        return {
            "answer": "",
            "source": "no_cache",
            "error": "no cache loaded; set KG_EDGES and POST /reload",
            "ctx": {"seeds": [], "paths": []},
        }

    seeds = detect_seeds_from_question(req.question, caches.nid2i.keys())
    paths = shortest_paths_multi_seed(
        caches,
        seed_ids=seeds,
        topk_paths=req.topk_paths,
        max_hops=req.max_hops,
        neighbor_expand=req.neighbor_expand,
        allowed_relation_names=req.allowed_relations,
    )

    ctx = {
        "seeds": seeds,
        "paths": paths,
        "contexts": [],
        "local_facts": [],
    }

    if dry_run:
        return {"answer": "", "source": "dry_run", "ctx": ctx, "note": "dry_run (LLM skipped)"}

    try:
        model = os.environ.get("OLLAMA_MODEL", _state["model_name"])
        answer = llm.generate_answer(model, req.question, paths)
        return {"answer": answer, "source": "llm", "ctx": ctx}
    except Exception as e:
        return {"answer": "", "source": "llm_error", "error": str(e), "ctx": ctx}
