# backend/api.py
from __future__ import annotations
import os, json, orjson, time
from typing import List, Dict, Tuple, Any
from fastapi import FastAPI, Request, Body, Query
from pydantic import BaseModel
import networkx as nx

# ---------------------------
# RERANKER (imports + helper)
# ---------------------------
try:
    from src.rerank import RapidBM25Reranker, RerankConfig
except Exception:
    RapidBM25Reranker = None
    RerankConfig = None

def _apply_rerank_if_enabled(question: str, result: dict, use_rerank: bool, rerank_mode: str, rerank_topk: int | None):
    if not use_rerank or not RapidBM25Reranker:
        return result
    holder = result.get("ctx") if isinstance(result, dict) and "ctx" in result else result
    paths = (holder or {}).get("paths") or []
    notes = (holder or {}).get("node_notes") or []
    if not paths:
        return result
    cfg = RerankConfig(mode=(rerank_mode or "hybrid"))
    rr = RapidBM25Reranker(cfg)
    order, scores = rr.rerank(question, notes, paths)
    new_paths = [paths[i] for i in order]
    new_scores = [scores[i] for i in order]
    if rerank_topk:
        k = int(rerank_topk)
        new_paths = new_paths[:k]
        new_scores = new_scores[:k]
    if holder is not result:
        result["ctx"]["paths"] = new_paths
        result["ctx"]["path_scores"] = new_scores
    else:
        result["paths"] = new_paths
        result["path_scores"] = new_scores
    meta = result.get("meta") or {}
    meta.update({"reranked": True, "rerank_mode": cfg.mode})
    result["meta"] = meta
    return result

# ---------------------------
# MODELS
# ---------------------------
class AskRequest(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 2
    use_llm: bool = True
    answer_style: str = "concise"  # "concise" | "verbose"
    explain: bool = True
    # Reranker knobs
    use_rerank: bool = True
    rerank_mode: str = "hybrid"    # "bm25" | "fuzz" | "hybrid"
    rerank_topk: int | None = None

# ---------------------------
# APP & STATE
# ---------------------------
app = FastAPI(title="Graph RAG Sprint API", version="0.1.0")
START_TIME = time.time()

class GraphState:
    def __init__(self):
        self.G: nx.DiGraph | None = None
        self.node_notes: Dict[str, str] = {}
        self.edges_path = os.getenv("EDGES_TSV", "./data/edges.tsv")
        self.notes_path = os.getenv("NODE_TEXTS", "./data/node_texts.jsonl")

STATE = GraphState()

# ---------------------------
# UTIL
# ---------------------------
def _orjson_dumps(v: Any, *, default=None) -> str:
    return orjson.dumps(v, default=default).decode("utf-8")

def load_graph() -> dict:
    """Load edges.tsv (tab: h, r, t) and node_texts.jsonl ({id,text})."""
    G = nx.DiGraph()
    if os.path.exists(STATE.edges_path):
        with open(STATE.edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split("\t")
                if len(parts) < 3: continue
                h, r, t = parts[0], parts[1], parts[2]
                G.add_node(h); G.add_node(t)
                G.add_edge(h, t, r=r)
    notes = {}
    if os.path.exists(STATE.notes_path):
        with open(STATE.notes_path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                try:
                    obj = json.loads(ln)
                    nid = str(obj.get("id") or obj.get("node") or obj.get("name") or "")
                    txt = str(obj.get("text") or obj.get("note") or "")
                    if nid: notes[nid] = txt
                except Exception:
                    continue
    STATE.G = G
    STATE.node_notes = notes
    return {"status": "ok", "edges": G.number_of_edges(), "nodes": G.number_of_nodes()}

def _match_nodes_by_question(q: str) -> List[str]:
    ql = (q or "").lower()
    cand = []
    if not STATE.G: return cand
    for n in STATE.G.nodes():
        if str(n).lower() in ql:
            cand.append(n)
    if not cand:
        cand = list(STATE.G.nodes())[:5]
    return sorted(set(cand))

def _k_shortest_paths(G: nx.DiGraph, src: str, dst: str, k: int, max_hops: int) -> List[List[dict]]:
    """Yield up to k simple paths (<= max_hops), as dict triples (h,r,t)."""
    paths: List[List[dict]] = []
    try:
        for p in nx.all_simple_paths(G, src, dst, cutoff=max_hops):
            triples = []
            for i in range(len(p)-1):
                u, v = p[i], p[i+1]
                r = G[u][v].get("r", "")
                triples.append({"h": u, "r": r, "t": v})
            paths.append(triples)
            if len(paths) >= k: break
    except nx.NetworkXNoPath:
        pass
    return paths

def _expand_neighbors(G: nx.DiGraph, nodes: List[str], hops: int) -> Dict[str, List[tuple]]:
    ctx = {}
    if hops <= 0: return ctx
    for n in nodes:
        local = []
        for _, v, data in G.out_edges(n, data=True):
            local.append((n, data.get("r",""), v))
        for u, _, data in G.in_edges(n, data=True):
            local.append((u, data.get("r",""), n))
        ctx[n] = local[:20]
    return ctx

def _build_answer(question: str, paths: List[List[dict]]) -> str:
    if not paths:
        return "I don't know."
    p = paths[0]
    if any((step.get("r","").lower() == "coauthored_with") for step in p):
        for step in p:
            if step.get("r","").lower() == "coauthored_with":
                return f"{step['h']} co-authored a paper with {step['t']}."
    ends = [p[0]["h"], p[-1]["t"]] if p else ["", ""]
    return f"I found a connection between {ends[0]} and {ends[1]}."

# ---------------------------
# LIFECYCLE
# ---------------------------
@app.on_event("startup")
def _startup():
    load_graph()

# ---------------------------
# ENDPOINTS
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - START_TIME)}

@app.get("/version")
def version():
    return {"name": "Graph RAG Sprint API", "version": app.version}

@app.post("/reload")
def reload_graph():
    info = load_graph()
    return {"status": "ok", "edges": STATE.edges_path, "texts": STATE.notes_path, **info}

@app.post("/ask")
async def ask(
    request: Request,
    dry_run: bool = Query(False, description="If true, skip answer text; return ctx only."),
    body: AskRequest = Body(...)
):
    G = STATE.G or nx.DiGraph()
    seeds = _match_nodes_by_question(body.question)

    # Collect paths between seeds until topk
    paths: List[List[dict]] = []
    if len(seeds) >= 2:
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                if i == j: continue
                found = _k_shortest_paths(G, seeds[i], seeds[j], k=body.topk_paths, max_hops=body.max_hops)
                for p in found:
                    paths.append(p)
                    if len(paths) >= body.topk_paths: break
                if len(paths) >= body.topk_paths: break

    contexts = _expand_neighbors(G, seeds, body.neighbor_expand)

    node_notes = []
    for s in seeds:
        note = STATE.node_notes.get(s)
        if note:
            node_notes.append(f"{s}: {note}")

    ctx = {
        "seeds": seeds,
        "paths": paths,
        "contexts": contexts,
        "local_facts": [],
        "node_notes": node_notes,
    }

    result = {
        "answer": None,
        "source": "llm" if body.use_llm else "graph",
        "ctx": ctx,
        "meta": {"answer_style": body.answer_style, "explain": body.explain}
    }

    # Rerank (hybrid by default)
    result = _apply_rerank_if_enabled(
        question=body.question,
        result=result,
        use_rerank=bool(body.use_rerank),
        rerank_mode=body.rerank_mode,
        rerank_topk=(body.rerank_topk or body.topk_paths)
    )

    if dry_run:
        return result

    holder = result.get("ctx", result)
    answer = _build_answer(body.question, holder.get("paths") or [])
    result["answer"] = answer

    if body.explain:
        result["evidence"] = {"seeds": seeds, "node_notes": node_notes[:10]}

    return result
