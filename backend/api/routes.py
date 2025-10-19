from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services import graph, rag

router = APIRouter()

class AskRequest(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 2
    dry_run: bool = False

@router.post("/reload")
def reload_data():
    state = graph.reload(edges="./data/edges.tsv", texts="./data/node_texts.jsonl")
    return {"status": "ok", **state}

@router.post("/ask")
def ask(req: AskRequest):
    if not graph.is_ready():
        raise HTTPException(status_code=400, detail="Graph not loaded. POST /reload first.")
    return rag.answer(req.dict())
