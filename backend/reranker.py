from __future__ import annotations
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import threading

_embed_lock = threading.Lock()
_embed_model = None

def _get_embedder():
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def build_candidate_texts(paths: List, node_texts: Dict[str, str]) -> List[str]:
    cand_texts = []
    for p in paths:
        # Accept either ["A","B","C"] or [(node,meta),...]
        if isinstance(p, list) and p and isinstance(p[0], (list, tuple)):
            node_ids = [n for n, _ in p]
        else:
            node_ids = p
        snippets = []
        for nid in node_ids:
            t = node_texts.get(str(nid)) or node_texts.get(nid) or ""
            if t:
                snippets.append(t[:500])
        cand_texts.append(" ; ".join(snippets) or " ".join(map(str, node_ids)))
    return cand_texts

def rerank_paths(
    question: str,
    paths: List,
    node_texts: Dict[str, str],
    strategy: str = "semantic",
    topk_paths: int = 5,
):
    if not paths:
        return paths

    cand_texts = build_candidate_texts(paths, node_texts)
    bm25 = BM25Okapi([t.split() for t in cand_texts])
    bm25_scores = bm25.get_scores(question.split())

    if strategy == "off":
        order = list(range(len(paths)))
    elif strategy == "bm25":
        order = sorted(range(len(paths)), key=lambda i: bm25_scores[i], reverse=True)
    else:
        model = _get_embedder()
        q_emb = model.encode([question], normalize_embeddings=True)
        c_emb = model.encode(cand_texts, normalize_embeddings=True)
        cos = util.cos_sim(q_emb, c_emb).cpu().numpy()[0]
        # lightly blend BM25 for robustness
        bmax = max(bm25_scores) if bm25_scores.size else 1.0
        combined = [(0.8 * float(cos[i]) + 0.2 * (bm25_scores[i] / (bmax + 1e-9)), i) for i in range(len(paths))]
        order = [i for _, i in sorted(combined, key=lambda z: z[0], reverse=True)]

    return [paths[i] for i in order[:topk_paths]]