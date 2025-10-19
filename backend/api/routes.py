from fastapi import APIRouter, Request
from pydantic import BaseModel
import os, httpx
from backend import graph_store

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
    from backend.graph_store import build_ctx
    dry_run = request.query_params.get("dry_run", "false").lower() == "true"
    ctx = build_ctx(payload.topk_paths, payload.max_hops, payload.neighbor_expand)
    if dry_run:
        return {"ctx": ctx}
    from backend.main import generate_answer as _gen
    return {"answer": _gen(payload.question, ctx, payload.model, 30), "ctx": ctx}

def _fetch_models(host: str) -> list[str]:
    url = f"{host.rstrip('/')}/api/tags"
    with httpx.Client() as c:
        r = c.get(url)
    if r.status_code == 200:
        data = r.json()
        return [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    return []

@router.get("/models")
def models():
    hosts = []
    env = os.environ.get("OLLAMA_HOST")
    if env:
        hosts.append(env)
    # dev-friendly fallback for Docker Desktop on Windows/macOS
    hosts.append("http://host.docker.internal:11434")
    seen = set()
    for h in hosts:
        if h in seen: 
            continue
        seen.add(h)
        try:
            names = _fetch_models(h)
            if names:
                return names
        except Exception:
            pass
    return []
