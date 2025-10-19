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
) -> List[PathTriples]:
    """
    Return up to topk simple paths (no repeated nodes) starting from any seed,
    up to max_hops edges, as lists of (src, rel, dst) triples. Traverses OUT-edges.
    """
    G: nx.MultiDiGraph = graph_store.graph()
    seeds = [s for s in seeds if G.has_node(s)]
    if not seeds or max_hops <= 0 or topk_paths <= 0:
        return []

    results: List[PathTriples] = []
    seen_endings: Set[Tuple[str, ...]] = set()  # to reduce duplicates

    for s in seeds:
        # stack items: (current_node, path_nodes, path_triples)
        stack: List[Tuple[str, List[str], PathTriples]] = [(s, [s], [])]
        while stack and len(results) < topk_paths * 4:  # search budget
            node, path_nodes, path_triples = stack.pop()
            if 0 < len(path_triples) <= max_hops:
                key = tuple(itertools.chain.from_iterable((u, v) for (u, _r, v) in path_triples))
                if key not in seen_endings:
                    seen_endings.add(key)
                    results.append(path_triples)
                    if len(results) >= topk_paths:
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

        if len(results) >= topk_paths:
            break

    # Prefer shorter paths; stable for determinism
    results.sort(key=lambda p: (len(p), p))
    return results[:topk_paths]


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
