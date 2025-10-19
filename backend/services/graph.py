# backend/services/graph.py
from __future__ import annotations
import json
import pathlib
import threading
from typing import Dict, Iterable, Tuple, Optional
import networkx as nx

Triple = Tuple[str, str, str]


class GraphStore:
    """
    Holds a directed multigraph with:
      - nodes: str ids, attrs: {'text': Optional[str]}
      - edges: (u, v, key) with attr {'rel': str}
    Safe for concurrent reads; writes are serialized.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._loaded_paths: Dict[str, str] = {}  # {'edges': str, 'nodes': str}

    # ---------- public API ----------
    def reload(self, edges_tsv: str, node_texts_jsonl: Optional[str] = None) -> Dict[str, int]:
        """
        Build graph from disk. Returns basic stats.
        """
        edges_path = _as_path(edges_tsv)
        nodes_path = _as_path(node_texts_jsonl) if node_texts_jsonl else None

        triples = list(_read_edges_tsv(edges_path))
        texts = _read_node_texts_jsonl(nodes_path) if nodes_path else {}

        g = _build_graph(triples, texts)

        with self._lock:
            self._g = g
            self._loaded_paths = {
                "edges": str(edges_path),
                "nodes": str(nodes_path) if nodes_path else ""
            }

        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "relations": len({d["rel"] for *_e, d in g.edges(data=True)}),
        }

    def graph(self) -> nx.MultiDiGraph:
        with self._lock:
            return self._g  # NOTE: treat as read-only outside

    def get_node_text(self, node: str) -> Optional[str]:
        with self._lock:
            if node in self._g:
                return self._g.nodes[node].get("text")
            return None

    def loaded_paths(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._loaded_paths)

    # ---------- convenience checks ----------
    def has_node(self, node: str) -> bool:
        with self._lock:
            return self._g.has_node(node)

    def degree(self, node: str) -> int:
        with self._lock:
            return self._g.degree(node)


# --------- helpers (pure functions) ----------
def _as_path(p: str | pathlib.Path | None) -> pathlib.Path:
    if p is None:
        raise ValueError("Path is None")
    return pathlib.Path(p).expanduser().resolve()


def _read_edges_tsv(path: pathlib.Path) -> Iterable[Triple]:
    """
    Yields (head, relation, tail). Skips blanks and comments (# ...).
    """
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{ln}: expected 3 tab-separated fields, got {len(parts)}")
            h, r, t = (p.strip() for p in parts)
            if not h or not r or not t:
                raise ValueError(f"{path}:{ln}: empty head/relation/tail")
            yield (h, r, t)


def _read_node_texts_jsonl(path: Optional[pathlib.Path]) -> Dict[str, str]:
    """
    Returns {node_id: text}. Lines with missing fields are ignored.
    """
    if not path:
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}") from e
            nid = (obj.get("id") or "").strip()
            txt = (obj.get("text") or "").strip()
            if nid:
                if txt:  # only keep non-empty text
                    out[nid] = txt
    return out


def _build_graph(triples: Iterable[Triple], node_text: Dict[str, str]) -> nx.MultiDiGraph:
    """
    Construct a MultiDiGraph from triples and attach node_text.
    """
    g = nx.MultiDiGraph()
    # add nodes first if text exists (so isolated nodes with text are preserved)
    for nid, txt in node_text.items():
        g.add_node(nid, text=txt)

    for (h, r, t) in triples:
        # ensure nodes exist
        if h not in g:
            g.add_node(h)
        if t not in g:
            g.add_node(t)
        # add edge with 'rel' attribute; MultiDiGraph keeps parallel edges (multi-relations)
        g.add_edge(h, t, rel=r)

    # fill missing 'text' with None explicitly (optional, but keeps schema tight)
    for n in g.nodes:
        g.nodes[n].setdefault("text", None)

    return g


# Singleton instance used by the app
graph_store = GraphStore()
