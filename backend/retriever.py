from __future__ import annotations
import json, pathlib, re
import networkx as nx
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

DATA = pathlib.Path(__file__).parents[1] / "data"
G = nx.MultiDiGraph()
NODE_TEXT: Dict[str, str] = {}
BM25 = None
DOCS: List[List[str]] = []
ID2IDX: Dict[str, int] = {}

_WORD = re.compile(r"[a-z0-9]+")

def _tok(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())

def load_graph(edges_path: str | None=None, node_texts_path: str | None=None):
    """Load TSV edges + JSONL node blurbs; build MultiDiGraph + BM25."""
    edges = edges_path or str(DATA / "kg_edges.tsv")
    texts = node_texts_path or str(DATA / "node_texts.jsonl")

    G.clear()
    if pathlib.Path(edges).exists():
        with open(edges, "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                s, d, r = parts[:3]
                w = float(parts[3]) if len(parts) > 3 else 1.0
                G.add_edge(s, d, key=r, relation=r, weight=w)
                G.add_node(s); G.add_node(d)

    global NODE_TEXT, DOCS, ID2IDX, BM25
    NODE_TEXT = {}
    if pathlib.Path(texts).exists():
        with open(texts, "r", encoding="utf-8-sig") as f:
            for line in f:
                obj = json.loads(line)
                nid, txt = obj["id"], obj.get("text","")
                NODE_TEXT[nid] = txt
                if not G.has_node(nid):
                    G.add_node(nid)

    # BM25 over node texts
    corpus = []
    ID2IDX = {}
    for i, (nid, txt) in enumerate(NODE_TEXT.items()):
        ID2IDX[nid] = i
        corpus.append(_tok(txt))
    DOCS = corpus
    BM25 = BM25Okapi(DOCS) if DOCS else None

def text_lookup(query: str, topk: int=10) -> List[str]:
    if not DOCS or BM25 is None:
        return []
    scores = BM25.get_scores(_tok(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    idx2id = {v:k for k,v in ID2IDX.items()}
    return [idx2id[i] for i in top_idx]

def content_overlap_lookup(query: str, topk: int=5) -> List[str]:
    """Very simple fallback: rank nodes by shared word count with query."""
    qset = set(_tok(query))
    if not qset or not NODE_TEXT:
        return []
    scored = []
    for nid, txt in NODE_TEXT.items():
        tset = set(_tok(txt))
        score = len(qset & tset)
        if score:
            scored.append((score, nid))
    scored.sort(reverse=True)
    return [nid for _s, nid in scored[:topk]]

def fuzzy_nodes(query: str, topk: int=5) -> List[str]:
    if not NODE_TEXT:
        return []
    choices = list(NODE_TEXT.keys())
    res = process.extract(query, choices, scorer=fuzz.WRatio, limit=topk)
    return [nid for nid, _score, _ in res]

def k_shortest_paths(src: str, dst: str, k: int=5, cutoff: int=3) -> List[List[Tuple[str,str,str]]]:
    paths = []
    try:
        gen = nx.shortest_simple_paths(G, source=src, target=dst, weight=None)
        for _ in range(k):
            p = next(gen)
            edges = []
            for u, v in zip(p[:-1], p[1:]):
                data = G.get_edge_data(u, v)
                if not data:
                    rel = "rel"
                else:
                    k0 = list(data.keys())[0]
                    rel = data[k0].get("relation", k0)
                edges.append((u, rel, v))
            if len(edges) <= cutoff:
                paths.append(edges)
            if len(paths) >= k:
                break
    except Exception:
        pass
    return paths

def expand_neighborhood(nid: str, hops: int=2, max_nodes: int=40):
    nodes = {nid}
    frontier = {nid}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            if not G.has_node(u):
                continue
            for v in G.successors(u):
                nxt.add(v)
            for v in G.predecessors(u):
                nxt.add(v)
        nodes |= nxt
        frontier = nxt
        if len(nodes) >= max_nodes:
            break
    sub = G.subgraph(nodes).copy()
    return sub

def retrieve(query: str, topk_paths: int=5, max_hops: int=3, neighbor_expand: int=2) -> Dict[str, Any]:
    """Seed via BM25 + fuzzy id + content overlap; paths among seeds; neighborhoods for ALL seeds."""
    candidates = []
    candidates += text_lookup(query, 8)
    candidates += content_overlap_lookup(query, 8)
    candidates += fuzzy_nodes(query, 5)
    raw_seeds = list(dict.fromkeys(candidates))
    seeds = [s for s in raw_seeds if G.has_node(s)][:8]

    path_bundle: List[List[Tuple[str,str,str]]] = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            pths = k_shortest_paths(seeds[i], seeds[j], k=topk_paths, cutoff=max_hops)
            if pths:
                path_bundle.extend(pths)

    local = []
    for nid in seeds[:4]:
        sub = expand_neighborhood(nid, hops=neighbor_expand, max_nodes=40)
        for (u, v, k_) in sub.edges(keys=True):
            rel = sub.get_edge_data(u, v)[k_].get("relation", k_)
            local.append((u, rel, v))

    def path_to_text(path):
        return " -> ".join([f"{u} -[{r}]-> {v}" for (u, r, v) in path])

    contexts = [path_to_text(p) for p in path_bundle]
    local_txt = [f"{u} -[{r}]-> {v}" for (u, r, v) in local]
    contexts = list(dict.fromkeys(contexts))[:topk_paths]
    local_txt = list(dict.fromkeys(local_txt))[:80]

    node_notes = []
    for nid in seeds[:10]:
        t = NODE_TEXT.get(nid, "")
        if t:
            node_notes.append(f"{nid}: {t}")

    return {
        "seeds": seeds,
        "paths": path_bundle,
        "contexts": contexts,
        "local_facts": local_txt,
        "node_notes": node_notes
    }
