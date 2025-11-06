from __future__ import annotations
import os, json, gzip, pathlib
from typing import Dict, List, Tuple, Iterable

import numpy as np
from scipy import sparse  # pip install scipy

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def _iter_lines(p: pathlib.Path) -> Iterable[str]:
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
            for line in f: yield line
    else:
        with p.open("r", encoding="utf-8", newline="") as f:
            for line in f: yield line

def read_edges(path: pathlib.Path) -> Tuple[List[str], List[str], List[str]]:
    H,R,T = [],[],[]
    for raw in _iter_lines(path):
        line = raw.strip()
        if not line: continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3: continue
        h,r,t = parts[:3]
        H.append(h); R.append(r); T.append(t)
    return H,R,T

def build_id_maps(nodes: List[str]):
    uniq = list(dict.fromkeys(nodes))  # preserve order
    return {n:i for i,n in enumerate(uniq)}, uniq

def build_csr(heads_i: List[int], tails_i: List[int], n: int):
    # undirected CSR for fast BFS
    rows = np.array(heads_i + tails_i, dtype=np.int32)
    cols = np.array(tails_i + heads_i, dtype=np.int32)
    data = np.ones_like(rows, dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

def main(edges_path: str, texts_path: str | None):
    EP = pathlib.Path(edges_path)
    TP = pathlib.Path(texts_path) if texts_path else None

    print(f"[cache] reading edges: {EP}")
    H,R,T = read_edges(EP)
    nid2i, i2nid = build_id_maps(list(dict.fromkeys(H+T)))
    heads_i = [nid2i[h] for h in H]
    tails_i = [nid2i[t] for t in T]
    A = build_csr(heads_i, tails_i, n=len(i2nid))

    # Save graph cache
    np.save(CACHE / "i2nid.npy", np.array(i2nid, dtype=object))
    np.savez_compressed(CACHE / "edges_it.npz",
                        heads=np.array(heads_i, dtype=np.int32),
                        tails=np.array(tails_i, dtype=np.int32))
    sparse.save_npz(str(CACHE / "adj_csr.npz"), A)

    # Save texts aligned to ids (optional)
    texts_i = []
    if TP and TP.exists():
        with TP.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw: continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                nid = obj.get("id") or obj.get("node_id") or obj.get("label") or obj.get("name")
                txt = obj.get("text") or obj.get("description") or obj.get("blurb") or ""
                if nid in nid2i and txt:
                    texts_i.append((nid2i[nid], txt))
    np.save(CACHE / "texts_idx.npy", np.array(texts_i, dtype=object))

    print(f"[cache] done |V|={len(i2nid)} |E|={len(heads_i)} |texts={len(texts_i)}")
    print(f"[cache] wrote to: {CACHE}")

if __name__ == "__main__":
    edges = os.getenv("KG_EDGES", str(DATA / "kg_edges.tsv"))
    texts = os.getenv("KG_NODE_TEXTS", str(DATA / "node_texts.jsonl"))
    main(edges, texts)
