from fastapi import FastAPI
import os, time
from backend.api.routes import router
from backend.services.graph import graph_store

app = FastAPI(title="Graph RAG Sprint API", version="0.1.0")
app.include_router(router)

# --- version + health ---
START_TS = time.time()

@app.get("/version")
def version():
    return {"name": "Graph RAG Sprint API", "version": app.version}

@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - START_TS)}

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
