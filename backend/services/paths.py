from __future__ import annotations
from typing import Iterable, List, Dict, Set, Tuple
import itertools
import networkx as nx

from backend.services.graph import graph_store

Triple = Tuple[str, str, str]
PathTriples = List[Triple]

def find_paths(
    seeds: Iterable[str],
    max_hops: int = 3,
    topk_paths: int = 5,
    use_gnn: bool = False,
) -> List[Dict[str, any]]:
    """
    Return up to topk simple paths (no repeated nodes) starting from any seed,
    up to max_hops edges, as lists of (src, rel, dst) triples. Traverses OUT-edges.
    
    If use_gnn=True, paths are re-ranked using GNN-based scoring.
    """
    G: nx.MultiDiGraph = graph_store.graph()
    seeds = [s for s in seeds if G.has_node(s)]
    if not seeds or max_hops <= 0 or topk_paths <= 0:
        return []

    results: List[PathTriples] = []
    seen_endings: Set[Tuple[str, ...]] = set()  # to reduce duplicates

    # Search for candidate paths (recall phase)
    search_limit = topk_paths * 4 if use_gnn else topk_paths * 2
    
    for s in seeds:
        # stack items: (current_node, path_nodes, path_triples)
        stack: List[Tuple[str, List[str], PathTriples]] = [(s, [s], [])]
        while stack and len(results) < search_limit:
            node, path_nodes, path_triples = stack.pop()
            if 0 < len(path_triples) <= max_hops:
                key = tuple(itertools.chain.from_iterable((u, v) for (u, _r, v) in path_triples))
                if key not in seen_endings:
                    seen_endings.add(key)
                    results.append(path_triples)
                    if len(results) >= search_limit:
                        break

            if len(path_triples) >= max_hops:
                continue

            # Traverse outgoing multi-edges
            for _, v, edata in G.out_edges(node, data=True):
                if v in path_nodes:  # simple paths only
                    continue
                rel = edata.get("rel", "")
                new_triple: Triple = (node, rel, v)
                stack.append((v, path_nodes + [v], path_triples + [new_triple]))

        if len(results) >= search_limit:
            break

    # Format paths with metadata
    formatted_paths = []
    for path_triples in results:
        path_dict = {
            "triples": [{"src": src, "relation": rel, "dst": dst} for src, rel, dst in path_triples],
            "length": len(path_triples),
            "text": _format_path_text(path_triples),
        }
        formatted_paths.append(path_dict)

    # GNN re-ranking (precision phase)
    if use_gnn:
        try:
            from backend.services.gnn_retriever import get_gnn_retriever
            gnn = get_gnn_retriever()
            if gnn is not None:
                gnn_scores = gnn.score_paths(formatted_paths)
                # Combine with length penalty (prefer shorter paths)
                for i, path in enumerate(formatted_paths):
                    length_penalty = 1.0 / (1.0 + path["length"])
                    path["gnn_score"] = gnn_scores[i]
                    path["final_score"] = 0.7 * gnn_scores[i] + 0.3 * length_penalty
                
                # Sort by GNN score
                formatted_paths.sort(key=lambda p: p.get("final_score", 0), reverse=True)
        except Exception as e:
            print(f"⚠️ GNN scoring failed: {e}")

    # Return top-k
    return formatted_paths[:topk_paths]


def _format_path_text(path_triples: PathTriples) -> str:
    """Format path as readable text"""
    parts = []
    for src, rel, dst in path_triples:
        # Get node text if available
        src_text = graph_store.get_text(src) or src
        dst_text = graph_store.get_text(dst) or dst
        parts.append(f"{src_text} —[{rel}]→ {dst_text}")
    return " | ".join(parts) if parts else ""


def neighborhood(
    nodes: Iterable[str],
    hops: int = 1,
    limit_per_node: int = 25,
) -> Set[str]:
    """
    Undirected neighborhood expansion around given nodes (to pull extra context).
    """
    G: nx.MultiDiGraph = graph_store.graph()
    visited: Set[str] = set()
    frontier: Set[str] = set(n for n in nodes if G.has_node(n))
    for _ in range(max(hops, 0)):
        next_frontier: Set[str] = set()
        for u in frontier:
            if u in visited:
                continue
            visited.add(u)
            # undirected neighbors (in + out)
            nbrs = set(G.predecessors(u)) | set(G.successors(u))
            # mild cap per layer
            for v in itertools.islice(nbrs, limit_per_node):
                if v not in visited:
                    next_frontier.add(v)
        frontier = next_frontier
    return visited | frontier
