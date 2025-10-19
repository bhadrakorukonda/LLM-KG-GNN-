from fastapi import APIRouter, Request
from pydantic import BaseModel
import os
import httpx

from backend import graph_store
from backend import main as backend_main  # for generate_answer monkeypatch in tests

router = APIRouter()

class AskPayload(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 1
    use_rerank: bool = False
    model: str = "ollama/llama3"

@router.post("/ask")
async def ask(request: Request, payload: AskPayload):
    dry_run = request.query_params.get("dry_run", "false").lower() == "true"
    ctx = graph_store.build_ctx(
        topk_paths=payload.topk_paths,
        max_hops=payload.max_hops,
        neighbor_expand=payload.neighbor_expand,
    )
    if dry_run:
        # Dry-run must NOT include "answer"
        return {"ctx": ctx}

    # call generate_answer via backend_main so tests can monkeypatch it
    answer = backend_main.generate_answer(
        question=payload.question,
        ctx=ctx,
        model=payload.model,
        timeout_s=30,
    )
    return {"answer": answer, "ctx": ctx}

@router.get("/models")
def models():
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/tags"
    try:
        with httpx.Client() as c:
            resp = c.get(url)  # tests stub this
        if resp.status_code == 200:
            data = resp.json()
            # Normalize to a plain list of model names, e.g., ["llama3"]
            names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
            return names
    except Exception:
        pass
    return []
    
class ReloadPayload(BaseModel):
    edges: str
    texts: str

@router.post("/reload")
def reload(payload: ReloadPayload):
    stats = graph_store.reload(payload.edges, payload.texts)
    return {"status": "ok", "edges": payload.edges, "texts": payload.texts, **stats}
