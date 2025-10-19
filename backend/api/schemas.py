from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel


class AskIn(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 1
    use_rerank: bool = True
    rerank_mode: str = "hybrid"
    model: str = "ollama/llama3"


class AskOut(BaseModel):
    question: str
    answer: Optional[str] = None
    ctx: Dict[str, Any]
    model: str


