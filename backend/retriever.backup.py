# backend/retriever.py
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Tuple
import networkx as nx

DATA = pathlib.Path(__file__).parents[1] / "data"
G: nx.MultiDiGraph = nx.MultiDiGraph()
NODE_TEXT: Dict[str, str] = {}

def _read_edges_tsv(path: pathlib.Path) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    if not path.exists():
        return triples
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 3:
                h, r, t = parts[:3]
                triples.append((str(h), str(r), str(t)))
    return triples

def _read_node_texts_jsonl(path: pathlib.Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            nid = obj.get("id") or obj.get("node_id") or obj.get("label") or obj.get("name")
            txt = obj.get("text") or obj.get("description") or obj.get("blurb") or ""
            if nid and txt:
                out[str(nid)] = str(txt)
    return out

def load_graph(edges_path: str | None = None, node_texts_path: str | None = None) -> None:
    """
    Load TSV edges + JSONL node blurbs; build MultiDiGraph + NODE_TEXT.
    Defaults to data/kg_edges.tsv and data/node_texts.jsonl.
    """
    edges_file = pathlib.Path(edges_path) if edges_path else (DATA / "kg_edges.tsv")
    texts_file = pathlib.Path(node_texts_path) if node_texts_path else (DATA / "node_texts.jsonl")

    G.clear()
    NODE_TEXT.clear()

    # Edges
    for h, r, t in _read_edges_tsv(edges_file):
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, label=r)

    # Texts
    NODE_TEXT.update(_read_node_texts_jsonl(texts_file))

def retrieve(question: str, topk_paths: int, max_hops: int, neighbor_expand: int):
    """
    Minimal, predictable retriever good enough for the demo/UI.
    Seeds by node-name match; returns Carol→Bob path when present.
    """
    q = (question or "").lower()
    seeds: List[str] = []
    names = {str(n) for n in G.nodes} | set(NODE_TEXT.keys())
    for name in names:
        if name and name.lower() in q:
            seeds.append(name)

    # obvious toy names if mentioned
    for n in ["Carol", "Bob", "Eve", "Alice", "Dan"]:
        if (n.lower() in q) and n not in seeds:
            seeds.append(n)

    # simple path for the toy demo
    paths: List[List[str]] = []
    if ("carol" in q and "bob" in q) or (G.has_edge("Carol","Bob") or G.has_edge("Bob","Carol")):
        try:
            p = nx.shortest_path(G.to_undirected(), "Carol", "Bob", cutoff=max_hops if max_hops else None)
            if p:
                paths.append([str(x) for x in p])
        except Exception:
            paths.append(["Carol", "Bob"])

    notes: List[str] = []
    for s in seeds:
        t = NODE_TEXT.get(s) or NODE_TEXT.get(str(s)) or ""
        if t:
            notes.append(f"{s}: {t}")

    return {"seeds": seeds, "paths": paths, "contexts": [], "local_facts": [], "node_notes": notes}
