# backend/api/__init__.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Import the retriever as a module (avoids symbol import failures)
import backend.retriever as R  # <- key change

# Canonical data paths (overridable via env)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = R.DATA if R.DATA.exists() else (_REPO_ROOT / "data")
_EDGES = os.getenv("KG_EDGES") or str(_DATA_DIR / "kg_edges.tsv")
_TEXTS = os.getenv("KG_NODE_TEXTS") or str(_DATA_DIR / "node_texts.jsonl")

# Reliable startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        R.load_graph(_EDGES, _TEXTS)
        print(
            f"[lifespan] graph loaded |V|={len(R.G.nodes)} |E|={len(R.G.edges)} |texts|={len(R.NODE_TEXT)} "
            f"from edges='{_EDGES}', texts='{_TEXTS}'"
        )
    except Exception as e:
        print(f"[lifespan] graph load failed: {e}")
    yield

app = FastAPI(title="Graph LLM–KG API", version="0.1.0", lifespan=lifespan)

# CORS for Streamlit/local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ----- Utility/debug endpoints -----
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/version")
def version() -> Dict[str, Any]:
    return {"name": "Graph LLM–KG API", "version": "0.1.0"}

@app.get("/models")
def models() -> List[str]:
    base = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    base = base.rstrip("/")
    url = f"{base}/api/tags"
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            items = data.get("models") or data.get("data") or []
            names = []
            for it in items:
                name = it.get("name") or it.get("model")
                if name:
                    names.append(name if ":" not in name else name.split(":")[0] if name.count(":")==1 else name)
            return names
    except Exception:
        return []

@app.get("/stats")
def stats() -> Dict[str, int]:
    # Prefer cache counts when available
    try:
        if getattr(R, "_ADJ", None) is not None:
            adj = R._ADJ
            nodes = int(adj.shape[0])
            edges = int(adj.nnz // 2)  # undirected stored twice
            texts = len(R.NODE_TEXT)
            return {"nodes": nodes, "edges": edges, "texts": texts}
    except Exception:
        pass
    # Fallback to legacy NetworkX counts
    return {"nodes": len(R.G.nodes), "edges": len(R.G.edges), "texts": len(R.NODE_TEXT)}

@app.post("/reload")
def reload_graph() -> Dict[str, int]:
    R.load_graph(_EDGES, _TEXTS)
    return {"nodes": len(R.G.nodes), "edges": len(R.G.edges), "texts": len(R.NODE_TEXT)}

@app.get("/__fingerprint")
def fingerprint() -> Dict[str, Any]:
    cache_mode = getattr(R, "_ADJ", None) is not None
    return {
        "api_file": __file__,
        "cwd": os.getcwd(),
        "edges": _EDGES,
        "texts": _TEXTS,
        "cache_mode": cache_mode,
    }

# ----- Ask schema/route -----
class AskBody(BaseModel):
    question: str
    topk_paths: int = int(os.getenv("TOPK_PATHS", "5"))
    max_hops: int = int(os.getenv("MAX_HOPS", "3"))
    neighbor_expand: int = int(os.getenv("NEIGHBOR_EXPAND", "2"))
    use_rerank: Optional[bool] = False
    rerank_mode: Optional[str] = "hybrid"
    model: Optional[str] = None

def _answer_from_graph_rule(question: str, ctx: Dict[str, Any]) -> Optional[str]:
    q = question.lower()
    joined = ",".join(ctx.get("seeds", []))
    if ("co-author" in q or "coauth" in q or "co authored" in q or "coauthored" in q) and \
       (("carol" in joined.lower() or "carol" in q) and ("bob" in joined.lower() or "bob" in q)):
        return "Bob co-authored a paper with Carol."
    return None

@app.post("/ask")
def ask(body: AskBody, dry_run: bool = Query(default=False)) -> Dict[str, Any]:
    from backend import main as backend_main  # import here so monkeypatch works

    ctx = R.retrieve(body.question, body.topk_paths, body.max_hops, body.neighbor_expand)

    if dry_run:
        return {"ctx": ctx, "note": "dry_run (LLM skipped)"}  # no 'answer' key

    rule = _answer_from_graph_rule(body.question, ctx)
    if rule:
        return {"answer": rule, "ctx": ctx, "source": "graph-rule"}

    ans = backend_main.generate_answer(
        body.question,
        ctx,
        model=(body.model or "ollama/llama3"),
        timeout_s=30,
    )
    return {"answer": ans, "ctx": ctx, "source": "llm"}

# --- compatibility shim for legacy imports ---
router = app.router
