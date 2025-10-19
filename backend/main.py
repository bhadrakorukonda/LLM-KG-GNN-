from fastapi import FastAPI, Body, Query
import os, time, json, hashlib, logging
from backend.api.routes import router
from backend import __version__
from backend.services.graph import graph_store
from backend.retrieval.core import find_paths as core_find_paths, rank_paths as core_rank_paths
from backend.models.provider import generate_answer
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Graph RAG Sprint API", version="0.1.0")

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("graph-rag")

# --- Tiny LRU cache ---
class _LRU:
    def __init__(self, cap: int = 128):
        self.cap = cap
        self._d = {}
        self._order = []
    def get(self, k):
        if k in self._d:
            self._order.remove(k)
            self._order.append(k)
            return self._d[k]
        return None
    def put(self, k, v):
        if k in self._d:
            self._d[k] = v
            self._order.remove(k)
            self._order.append(k)
            return
        if len(self._order) >= self.cap:
            old = self._order.pop(0)
            self._d.pop(old, None)
        self._d[k] = v
        self._order.append(k)
    def clear(self):
        self._d.clear(); self._order.clear()

_ASK_CACHE = _LRU(128)

# --- CORS (dev) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# --- version + health ---
START_TS = time.time()

@app.get("/version")
def version():
    return {"name": "Graph RAG Sprint API", "version": __version__}

@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - START_TS)}

# --- models endpoint ---
@app.get("/models")
def models():
    host = os.environ.get("OLLAMA_HOST")
    if host:
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=3.0, read=5.0)) as client:
                resp = client.get(host.rstrip("/") + "/api/tags")
                if resp.status_code == 200:
                    data = resp.json() or {}
                    items = data.get("models") or []
                    names = [m.get("name") for m in items if m.get("name")]
                    return names
        except Exception:
            pass
    return ["ollama/llama3","ollama/llama3.2:3B"]

def _log_ask(question, topk_paths, max_hops, neighbor_expand, use_rerank, rerank_mode, model, dry_run, ctx_out, t0, cache_hit: bool):
    try:
        qhash = hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:10]
        payload = {
            "event": "ask",
            "q": qhash,
            "params": {
                "topk_paths": topk_paths,
                "max_hops": max_hops,
                "neighbor_expand": neighbor_expand,
                "use_rerank": use_rerank,
                "rerank_mode": rerank_mode,
                "model": model,
                "dry_run": bool(dry_run),
            },
            "seeds": len(ctx_out.get("seeds", [])),
            "paths": len(ctx_out.get("paths", [])),
            "duration_ms": int((time.time() - t0) * 1000),
            "cache_hit": cache_hit,
        }
        logger.info(json.dumps(payload))
    except Exception:
        pass

# --- retrieval endpoint ---
@app.post("/ask")
def ask(
    question: str = Body(..., embed=True),
    topk_paths: int = Body(5),
    max_hops: int = Body(3),
    neighbor_expand: int = Body(1),
    use_rerank: bool = Body(True),
    rerank_mode: str = Body("hybrid"),
    model: str = Body("ollama/llama3"),
    dry_run: bool = Query(False),
):
    if graph_store.graph().number_of_nodes() == 0:
        return {"error": "Graph not loaded. POST /reload first."}

    ctx = core_find_paths(
        question=question,
        topk=topk_paths,
        max_hops=max_hops,
        neighbor_expand=neighbor_expand,
    )

    order, scores = core_rank_paths(
        question=question,
        paths=ctx["paths"],
        node_notes=ctx.get("node_notes") or [],
        mode=rerank_mode,
    ) if use_rerank else (list(range(len(ctx["paths"]))), [0.0]*len(ctx["paths"]))

    ranked_paths = [ctx["paths"][i] for i in order]
    # Cap sizes for production safety
    MAX_PATHS_RETURNED = 20
    MAX_NODE_NOTES = 50
    ranked_paths = ranked_paths[:MAX_PATHS_RETURNED]
    node_notes = (ctx.get("node_notes") or [])[:MAX_NODE_NOTES]
    scores_sorted = [scores[i] for i in order]
    ctx_out = {**ctx, "paths": ranked_paths, "path_scores": scores_sorted[:len(ranked_paths)], "node_notes": node_notes}

    # cache key
    key_tuple = (question, topk_paths, max_hops, neighbor_expand, use_rerank, rerank_mode, model, bool(dry_run))
    cache_key = hashlib.sha1(json.dumps(key_tuple, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    cached = _ASK_CACHE.get(cache_key)
    t0 = time.time()
    if cached is not None:
        cached = {**cached, "cache_hit": True}
        _log_ask(question, topk_paths, max_hops, neighbor_expand, use_rerank, rerank_mode, model, dry_run, ctx_out, t0, cache_hit=True)
        return cached

    if dry_run:
        out = {"question": question, "ctx": ctx_out, "model": model}
        _ASK_CACHE.put(cache_key, out)
        _log_ask(question, topk_paths, max_hops, neighbor_expand, use_rerank, rerank_mode, model, dry_run, ctx_out, t0, cache_hit=False)
        return out

    answer = generate_answer(question=question, ctx=ctx_out, model=model)
    out = {"question": question, "answer": answer, "ctx": ctx_out, "model": model}
    _ASK_CACHE.put(cache_key, out)
    _log_ask(question, topk_paths, max_hops, neighbor_expand, use_rerank, rerank_mode, model, dry_run, ctx_out, t0, cache_hit=False)
    return out

# --- auto-load graph on startup (kept from earlier) ---
@app.on_event("startup")
async def _startup_load_graph():
    edges = os.environ.get("GRAPH_EDGES_PATH", "./data/edges.tsv")
    nodes = os.environ.get("GRAPH_NODE_TEXTS_PATH", "./data/node_texts.jsonl")
    try:
        stats = graph_store.reload(edges, nodes)
        print(f"[startup] Graph loaded: {stats}")
    except Exception as e:
        print(f"[startup] Graph NOT loaded: {e}")
