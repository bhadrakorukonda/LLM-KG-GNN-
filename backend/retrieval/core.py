from __future__ import annotations
import os
from typing import Dict, List, Tuple, Any, Iterable

from backend.services.graph import graph_store
from backend.services.paths import find_paths as find_graph_paths, neighborhood

try:
    from src.rerank import RapidBM25Reranker, RerankConfig
except Exception:
    RapidBM25Reranker = None  # type: ignore
    RerankConfig = None  # type: ignore


# --------------------------- CONFIG ---------------------------
EDGES_TSV: str = os.getenv("GRAPH_EDGES_PATH", "./data/edges.tsv")
NODE_TEXTS_JSONL: str | None = os.getenv("GRAPH_NODE_TEXTS_PATH", "./data/node_texts.jsonl")


# --------------------------- API ---------------------------
def load_graph(edges_path: str | None = None, node_texts_path: str | None = None) -> Dict[str, int]:
    """
    Load the knowledge graph into the in-memory store.

    Parameters
    - edges_path: path to TSV with triples (head, relation, tail). If None, uses EDGES_TSV.
    - node_texts_path: path to JSONL with node text. If None, uses NODE_TEXTS_JSONL.

    Returns
    - Basic statistics about the loaded graph: number of nodes, edges, relations.
    """
    edges = edges_path or EDGES_TSV
    texts = node_texts_path if node_texts_path is not None else NODE_TEXTS_JSONL
    return graph_store.reload(edges, texts)


def find_paths(
    question: str,
    topk: int = 5,
    max_hops: int = 3,
    neighbor_expand: int = 1,
) -> Dict[str, Any]:
    """
    Retrieve graph paths and local neighborhood context seeded by entities
    matched from the natural-language question.

    Returns a dictionary with keys:
      - seeds: List[str]
      - paths: List[List[Tuple[str, str, str]]]
      - contexts: Dict[str, List[Tuple[str, str, str]]]
      - node_notes: List[str]  (if node text is present in the graph)
    """
    G = graph_store.graph()

    # very simple seed matching: include nodes whose id appears in the question
    ql = (question or "").lower()
    seeds: List[str] = []
    for n in G.nodes:
        try:
            nid = str(n)
        except Exception:
            continue
        if nid and nid.lower() in ql:
            seeds.append(nid)

    if not seeds:
        # fallback: sample first few nodes deterministically
        seeds = [str(n) for n in list(G.nodes)[:5]]

    path_triples: List[List[Tuple[str, str, str]]] = find_graph_paths(
        seeds=seeds,
        max_hops=max_hops,
        topk_paths=topk,
    )

    # neighborhood context for explainability
    contexts: Dict[str, List[Tuple[str, str, str]]] = {}
    if neighbor_expand and neighbor_expand > 0:
        around = neighborhood(nodes=seeds, hops=neighbor_expand, limit_per_node=25)
        # convert to simple edge triples around each seed (limited fanout when reading)
        for s in seeds:
            local: List[Tuple[str, str, str]] = []
            for _, v, data in G.out_edges(s, data=True):
                local.append((s, data.get("rel", ""), str(v)))
            for u, _, data in G.in_edges(s, data=True):
                local.append((str(u), data.get("rel", ""), s))
            contexts[s] = local[:20]

    # collect node notes if present
    node_notes: List[str] = []
    for s in seeds:
        txt = graph_store.get_node_text(s)
        if txt:
            node_notes.append(f"{s}: {txt}")

    return {
        "seeds": seeds,
        "paths": path_triples,
        "contexts": contexts,
        "node_notes": node_notes,
    }


def rank_paths(
    question: str,
    paths: List[List[Tuple[str, str, str]]],
    node_notes: List[str] | None = None,
    mode: str = "hybrid",
) -> Tuple[List[int], List[float]]:
    """
    Rerank candidate paths using BM25 and/or fuzzy scoring when available.

    Returns order indices and scores. If reranker dependencies are missing,
    returns a no-op ranking (original order with zero scores).
    """
    if RapidBM25Reranker is None or RerankConfig is None:
        order = list(range(len(paths)))
        scores = [0.0 for _ in paths]
        return order, scores

    cfg = RerankConfig(mode=mode)
    rr = RapidBM25Reranker(cfg)
    return rr.rerank(question, node_notes or [], paths)


