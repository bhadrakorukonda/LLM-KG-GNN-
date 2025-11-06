# backend/retriever.py
from __future__ import annotations
import os, gzip, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
from scipy.sparse import csr_matrix

# -------------------- Paths & Back-compat --------------------

# Repo root = two levels up from this file: backend/retriever.py -> backend/ -> repo/
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Expose DATA as a Path (api/__init__.py does: R.DATA.exists())
DATA = _REPO_ROOT / "data"

# Cache dir & files (use Path; expose str for np.save/np.load)
CACHE_DIR_PATH = DATA / "cache"
EDGES_CACHE = str(CACHE_DIR_PATH / "adj_csr.npz")
I2NID_CACHE = str(CACHE_DIR_PATH / "i2nid.npy")
TEXTS_IDX_CACHE = str(CACHE_DIR_PATH / "texts_idx.npy")

# --- cache presence helper (used by backend.server) ---
def has_cache() -> bool:
    try:
        return (
            os.path.exists(EDGES_CACHE)
            and os.path.exists(I2NID_CACHE)
            and os.path.exists(TEXTS_IDX_CACHE)
        )
    except Exception:
        return False

# --- shims expected by backend.api.__init__ ---
class _NodeView:
    def __init__(self, adj): self._adj = adj
    def __len__(self): return int(self._adj.shape[0])

class _EdgeView:
    def __init__(self, adj): self._adj = adj
    def __len__(self): return int(self._adj.nnz)

class _GraphShim:
    def __init__(self, adj):
        self.nodes = _NodeView(adj)
        self.edges = _EdgeView(adj)

# Globals that api/__init__.py expects
G = None           # will become _GraphShim(adj)
NODE_TEXT: List[str] = []  # list-like; len() must work

# -------------------- Types --------------------

@dataclass
class GraphCaches:
    adj: csr_matrix                 # CSR over node indices
    rel_ids: np.ndarray             # shape (nnz,) relation int id for each edge in adj.indices order
    nid2i: Dict[str,int]            # node-id string -> index
    i2nid: np.ndarray               # index -> node-id string
    rel2i: Dict[str,int]            # relation string -> id
    i2rel: List[str]                # id -> relation string
    text_by_i: List[str]            # index -> node text (may be '')

# -------------------- IO helpers --------------------

def _open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def _ensure_dir(p: Path | str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _load_edges(path: str) -> Iterable[Tuple[str,str,str]]:
    # TSV: src \t rel \t dst
    with _open_maybe_gz(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            parts=line.split("\t")
            if len(parts) < 3:
                continue
            yield parts[0], parts[1], parts[2]

def _load_texts_jsonl(path: str) -> Dict[str,str]:
    """jsonl lines like: {"id": "...", "text": "..."}"""
    out={}
    with _open_maybe_gz(path) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
                nid=str(obj.get("id") or obj.get("node_id") or obj.get("name") or "")
                if not nid:
                    continue
                out[nid]=str(obj.get("text") or obj.get("content") or obj.get("desc") or "")
            except Exception:
                continue
    return out

# -------------------- Cache build / load --------------------

def build_cache(edges_tsv: str, texts_jsonl: Optional[str], *_args, **_kwargs) -> GraphCaches:
    """
    Build CSR + aux caches from (tsv[.gz], jsonl[.gz]) and write to data/cache.
    """
    _ensure_dir(CACHE_DIR_PATH)

    # 1) Index nodes/relations
    nodes:set[str]=set()
    rels:set[str]=set()
    edges=list(_load_edges(edges_tsv))
    for s, r, o in edges:
        nodes.add(s); nodes.add(o); rels.add(r)

    i2nid=np.array(sorted(nodes), dtype=object)
    nid2i={nid:i for i,nid in enumerate(i2nid)}
    i2rel=sorted(list(rels))
    rel2i={r:i for i,r in enumerate(i2rel)}

    # 2) Build COO arrays
    rows=[]; cols=[]; rel_ids=[]
    for s, r, o in edges:
        si=nid2i[s]; oi=nid2i[o]
        rows.append(si); cols.append(oi); rel_ids.append(rel2i[r])

    rows=np.asarray(rows, dtype=np.int64)
    cols=np.asarray(cols, dtype=np.int64)
    rel_ids=np.asarray(rel_ids, dtype=np.int32)

    data=np.ones_like(rows, dtype=np.int8)
    n=len(i2nid)
    adj=csr_matrix((data,(rows,cols)), shape=(n,n), dtype=np.int8)

    # 3) Texts
    text_by_i=[""]*n
    texts_idx=np.full(n, -1, dtype=np.int32)
    if texts_jsonl and os.path.exists(texts_jsonl):
        nid2text=_load_texts_jsonl(texts_jsonl)
        for i,nid in enumerate(i2nid):
            if nid in nid2text:
                text_by_i[i]=nid2text[nid]
                texts_idx[i]=i  # placeholder "index" compatibility

    # 4) Persist
    np.save(I2NID_CACHE, i2nid)
    np.save(TEXTS_IDX_CACHE, texts_idx)
    np.savez_compressed(EDGES_CACHE, indptr=adj.indptr, indices=adj.indices, data=adj.data, shape=adj.shape, rel_ids=rel_ids)

    # Back-compat globals
    global G, NODE_TEXT
    G = _GraphShim(adj)
    NODE_TEXT = text_by_i

    return GraphCaches(adj=adj, rel_ids=rel_ids, nid2i=nid2i, i2nid=i2nid, rel2i=rel2i, i2rel=i2rel, text_by_i=text_by_i)

def load_graph(*_args, **_kwargs) -> GraphCaches:
    """
    Cache-first loader; tolerant to legacy caches (pickle/object arrays, missing rel_ids).
    Falls back to building from env if caches missing.
    Env:
      KG_EDGES=path/to/edges.tsv[.gz]
      KG_NODE_TEXTS=path/to/node_texts.jsonl[.gz]  (optional)
    """
    edges_cache = EDGES_CACHE
    if os.path.exists(edges_cache) and os.path.exists(I2NID_CACHE) and os.path.exists(TEXTS_IDX_CACHE):
        # NPZ: try safe first, then allow_pickle if legacy object arrays exist
        try:
            z = np.load(edges_cache, allow_pickle=False)
        except ValueError:
            z = np.load(edges_cache, allow_pickle=True)

        indptr = z["indptr"]; indices = z["indices"]; data = z["data"]
        shape = tuple(z["shape"])  # type: ignore
        rel_ids = z["rel_ids"] if "rel_ids" in z.files else np.full(indices.shape[0], -1, dtype=np.int32)

        adj = csr_matrix((data, indices, indptr), shape=shape)

        # i2nid was intentionally saved as object dtype -> needs pickle
        i2nid = np.load(I2NID_CACHE, allow_pickle=True)
        nid2i = {nid: i for i, nid in enumerate(i2nid)}

        # texts index (we don’t deref here)
        _ = np.load(TEXTS_IDX_CACHE, allow_pickle=False)
        text_by_i = [""] * len(i2nid)

        # Back-compat globals
        global G, NODE_TEXT
        G = _GraphShim(adj)
        NODE_TEXT = text_by_i

        return GraphCaches(
            adj=adj,
            rel_ids=rel_ids,
            nid2i=nid2i,
            i2nid=i2nid,
            rel2i={}, i2rel=[],
            text_by_i=text_by_i
        )

    # No cache -> build from env
    edges = os.environ.get("KG_EDGES", "").strip()
    texts = os.environ.get("KG_NODE_TEXTS", "").strip() or None
    if not edges:
        raise RuntimeError("KG_EDGES not set and cache missing; cannot load graph.")
    return build_cache(edges, texts)

# -------------------- Retrieval --------------------

def _neighbors_cap(adj: csr_matrix, node: int, cap: int) -> np.ndarray:
    start, end = adj.indptr[node], adj.indptr[node+1]
    neigh = adj.indices[start:end]
    if cap <= 0 or len(neigh) <= cap:
        return neigh
    # Stable deterministic cap
    return neigh[:cap]

def _edge_rel_slice(rel_ids: np.ndarray, indptr: np.ndarray, node: int) -> np.ndarray:
    s, e = indptr[node], indptr[node+1]
    return rel_ids[s:e]

def _match_allowed(allowed: Optional[set], rels_for_row: np.ndarray) -> np.ndarray:
    if not allowed:
        return np.ones_like(rels_for_row, dtype=bool)
    # keep neighbor positions whose relation id is in allowed (ids, not names)
    allowed_ids=np.array(sorted(list(allowed)), dtype=np.int64)
    return np.isin(rels_for_row, allowed_ids)

def shortest_paths_multi_seed(
    caches: GraphCaches,
    seed_ids: List[str],
    topk_paths: int = 5,
    max_hops: int = 3,
    neighbor_expand: int = 2,
    allowed_relation_names: Optional[List[str]] = None
) -> List[List[str]]:
    """
    Multi-seed BFS producing up to topk shortest paths (as node-id sequences).
    If a single entity is present in the question, we still expand from it (self paths allowed).
    Optional relation filter (by relation names).
    """
    adj=caches.adj
    indptr=adj.indptr
    seeds=[caches.nid2i[s] for s in seed_ids if s in caches.nid2i]
    if not seeds:
        return []

    # relation filter map -> ids if we have a rel2i
    allowed_ids: Optional[set] = None
    if allowed_relation_names and caches.rel2i:
        allowed_ids={caches.rel2i[r] for r in allowed_relation_names if r in caches.rel2i}

    # BFS frontier per seed
    from collections import deque
    results: List[List[int]]=[]
    seen_pairs=set()

    for sidx in seeds:
        q=deque()
        q.append([sidx])
        while q and len(results) < topk_paths:
            path=q.popleft()
            u=path[-1]
            # expand neighbors with fan-out cap and (optional) relation filter
            neigh = _neighbors_cap(adj, u, neighbor_expand)
            relslice = _edge_rel_slice(caches.rel_ids, indptr, u)
            mask = _match_allowed(allowed_ids, relslice) if allowed_ids is not None else np.ones_like(relslice, dtype=bool)
            neigh = neigh[mask]
            for v in neigh:
                if v==u:
                    continue
                newp=path+[v]
                key=(newp[0], newp[-1], len(newp))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                q.append(newp)
                if 1 < len(newp) <= (max_hops+1):
                    results.append(newp)
                if len(results) >= topk_paths:
                    break

    # map to node-id strings
    out=[]
    for p in results[:topk_paths]:
        out.append([str(caches.i2nid[i]) for i in p])
    return out

# -------------------- Helpers for API --------------------

_SEED_REGEX=re.compile(r"[A-Z][a-zA-Z0-9_]+")  # naive default: capitalized tokens

def detect_seeds_from_question(q: str, known_node_ids: Iterable[str]) -> List[str]:
    candidates=set(_SEED_REGEX.findall(q))
    known=set(known_node_ids)
    return [c for c in candidates if c in known]
