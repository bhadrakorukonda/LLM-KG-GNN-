# src/rerank.py
from dataclasses import dataclass
from typing import List, Tuple
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

def _path_to_text(p):
    parts = []
    for step in p or []:
        if isinstance(step, dict):
            h = step.get("h") or step.get("src")
            r = step.get("r") or step.get("rel")
            t = step.get("t") or step.get("dst")
        else:
            h, r, t = step
        parts.append(f"{h} -[{r}]-> {t}")
    return "  ;  ".join(parts)

def _tokenize(s: str) -> List[str]:
    # simple, robust tokenization for BM25
    return (s or "")\
        .lower()\
        .replace("[", " ")\
        .replace("]", " ")\
        .replace("->", " ")\
        .replace("-", " ")\
        .split()

@dataclass
class RerankConfig:
    mode: str = "hybrid"        # "bm25" | "fuzz" | "hybrid"
    w_bm25: float = 0.6
    w_fuzz: float = 0.4

class RapidBM25Reranker:
    """
    Reranks paths using a hybrid BM25 + RapidFuzz score.
    Inputs:
      question: str
      node_notes: List[str]
      paths: List[List[step]]   (step is dict {'h','r','t'} or tuple (h,r,t))
    Returns:
      (order_indices, scores)
    """
    def __init__(self, cfg: RerankConfig = RerankConfig()):
        self.cfg = cfg

    def rerank(self, question: str, node_notes: List[str], paths: List[List]) -> Tuple[List[int], List[float]]:
        path_texts = [_path_to_text(p) for p in (paths or [])]
        docs = [f"{path_texts[i]}  ||  {'  |  '.join(node_notes or [])}" for i in range(len(path_texts))]

        tokenized = [_tokenize(d) for d in docs] if docs else []
        bm25 = BM25Okapi(tokenized) if tokenized else None
        bm_scores = bm25.get_scores(_tokenize(question)).tolist() if bm25 else [0.0]*len(paths)

        fuzz_scores = [fuzz.token_set_ratio(question or "", d) / 100.0 for d in docs]

        if self.cfg.mode == "bm25":
            scores = bm_scores
        elif self.cfg.mode == "fuzz":
            scores = fuzz_scores
        else:  # hybrid
            scores = [self.cfg.w_bm25*b + self.cfg.w_fuzz*f for b, f in zip(bm_scores, fuzz_scores)]

        order = sorted(range(len(paths)), key=lambda i: scores[i], reverse=True)
        return order, scores
