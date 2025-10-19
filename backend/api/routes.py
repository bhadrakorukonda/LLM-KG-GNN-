from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.services.graph import graph_store
from backend.services import rag

router = APIRouter()

class ReloadIn(BaseModel):
    edges: str = "./data/edges.tsv"
    node_texts: str | None = "./data/node_texts.jsonl"

class AskRequest(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 1
    dry_run: bool = False
    # LLM toggles
    use_llm: bool = False
    llm_model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 256
    explain: bool = False         # NEW: return short justification
    answer_style: str | None = None  # NEW: "short" | "sentence" | "bullet"

@router.post("/reload")
def reload_data(body: ReloadIn):
    try:
        stats = graph_store.reload(body.edges, body.node_texts)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reload failed: {e}")

@router.post("/ask")
async def ask(request: Request):
    # Merge JSON body + query params (query takes precedence)
    body = await request.json()
    params = dict(request.query_params)
    merged = {**body, **params}

    # Coerce bool-like strings from query
    for k in ["dry_run","use_llm","explain"]:
        if k in merged and isinstance(merged[k], str):
            merged[k] = merged[k].lower() in ("1","true","yes","y","on")

    # Coerce numerics if passed via query
    for k in ["topk_paths","max_hops","neighbor_expand","max_tokens","temperature"]:
        if k in merged and isinstance(merged[k], str):
            try:
                merged[k] = int(merged[k]) if k != "temperature" else float(merged[k])
            except:
                pass

    G = graph_store.graph()
    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="Graph not loaded. POST /reload first.")
    return rag.answer(merged)
