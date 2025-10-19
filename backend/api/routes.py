from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.retrieval.core import load_graph
import os

router = APIRouter()


class ReloadIn(BaseModel):
    edges: str = "./data/edges.tsv"
    node_texts: str | None = "./data/node_texts.jsonl"


@router.post("/reload")
def reload_data(body: ReloadIn | None = None):
    try:
        edges = body.edges if body else os.environ.get("GRAPH_EDGES_PATH", "./data/edges.tsv")
        notes = body.node_texts if body and body.node_texts is not None else os.environ.get("GRAPH_NODE_TEXTS_PATH", "./data/node_texts.jsonl")
        stats = load_graph(edges, notes)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reload failed: {e}")
