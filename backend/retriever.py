from __future__ import annotations
import os, json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

# --- Paths exposed for backend.api import-time usage ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).resolve()

# Default filenames (the API passes explicit paths to load_graph on startup/reload)
KG_EDGES = os.environ.get("KG_EDGES", str(DATA / "kg_edges.tsv"))
KG_NODE_TEXTS = os.environ.get("KG_NODE_TEXTS", str(DATA / "node_texts.jsonl"))

# --- In-memory graph ---
_GRAPH_LOADED = False
_ADJ: Dict[str, List[Tuple[str,str]]] = defaultdict(list)   # node -> list[(rel, neighbor)]
_REV: Dict[str, List[Tuple[str,str]]] = defaultdict(list)   # reverse edges
_NODE_TEXT: Dict[str, str] = {}

def _reset_graph():
    global _GRAPH_LOADED, _ADJ, _REV, _NODE_TEXT
    _GRAPH_LOADED = False
    _ADJ = defaultdict(list)
    _REV = defaultdict(list)
    _NODE_TEXT = {}

def _ensure_graph():
    # Lazy load using current KG_EDGES / KG_NODE_TEXTS
    global _GRAPH_LOADED
    if _GRAPH_LOADED:
        return
    _load_from_files(KG_EDGES, KG_NODE_TEXTS)
    _GRAPH_LOADED = True

def _load_from_files(edges_path: str, texts_path: str):
    # tolerate BOMs
    if os.path.exists(edges_path):
        with open(edges_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"): continue
                parts=line.split("\t")
                if len(parts) < 3: continue
                h, r, t = parts[0], parts[1], parts[2]
                _ADJ[h].append((r, t))
                _REV[t].append((r, h))
    if os.path.exists(texts_path):
        with open(texts_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    rec=json.loads(line)
                    nid=rec.get("id")
                    txt=rec.get("text") or rec.get("desc") or ""
                    if nid: _NODE_TEXT[nid]=txt
                except Exception:
                    pass

def _neighbors(u: str, rel_filter: Optional[Set[str]]):
    for r, v in _ADJ.get(u, []):
        if (rel_filter is None) or (r in rel_filter): yield (r, v)
    for r, v in _REV.get(u, []):
        if (rel_filter is None) or (r in rel_filter): yield (r, v)

def _enumerate_paths(seeds: List[str], max_hops: int, topk_paths: int, rel_filter: Optional[Set[str]]):
    paths: List[List[str]] = []
    def dfs(path: List[str], depth: int):
        if len(paths) >= topk_paths: return
        if depth >= max_hops: return
        u = path[-1]
        for r, v in _neighbors(u, rel_filter):
            if v in path:  # avoid cycles
                continue
            new_path = path + [v]
            paths.append(new_path)
            if len(paths) >= topk_paths: return
            dfs(new_path, depth + 1)
            if len(paths) >= topk_paths: return
    for s in seeds:
        if len(paths) >= topk_paths: break
        dfs([s], 0)
    return paths

def graph_counts() -> Dict[str, int]:
    # nodes from keys in both directions; edges from forward adj; texts from notes
    nodes = set(_ADJ.keys()) | set(_REV.keys())
    edge_count = sum(len(v) for v in _ADJ.values())
    text_count = len(_NODE_TEXT)
    return {"nodes": len(nodes), "edges": edge_count, "texts": text_count}

def load_graph(edges_path: str, texts_path: str) -> Dict[str, int]:
    """
    API calls this on startup and /reload.
    It must reset globals, load the graph, and return a dict with counts:
      {"nodes": N, "edges": M, "texts": T}
    """
    global KG_EDGES, KG_NODE_TEXTS
    # update active files
    KG_EDGES = edges_path
    KG_NODE_TEXTS = texts_path
    # clear then load
    _reset_graph()
    _load_from_files(KG_EDGES, KG_NODE_TEXTS)
    # mark loaded
    global _GRAPH_LOADED
    _GRAPH_LOADED = True
    return graph_counts()

def retrieve(question: str, topk_paths: int, max_hops: int, neighbor_expand: int,
             seeds: Optional[List[str]] = None, relation_filter: Optional[List[str]] = None):
    """
    Minimal retriever used by backend.api.__init__.ask():
    returns {seeds, paths, contexts, local_facts, node_notes}
    """
    _ensure_graph()

    if not seeds:
        seeds = []
        toks = set((question or "").replace("?"," ").replace(","," ").split())
        for nid in _NODE_TEXT.keys():
            if nid in toks: seeds.append(nid)
        if not seeds:
            # fallback to hubs
            seeds = sorted(_ADJ.keys(), key=lambda k: len(_ADJ[k]) + len(_REV[k]), reverse=True)[:3]

    rel_set = set(relation_filter) if relation_filter else None
    paths = _enumerate_paths(seeds, max_hops=max_hops, topk_paths=topk_paths, rel_filter=rel_set)

    node_notes = [(f"{nid}: {_NODE_TEXT.get(nid,'')}").rstrip() for nid in seeds]

    return {
        "seeds": seeds,
        "paths": paths,
        "contexts": [],
        "local_facts": [],
        "node_notes": node_notes,
    }
