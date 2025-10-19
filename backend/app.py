import os, json
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import requests

# ---------- Retriever (use your real one if present) ----------
try:
    from .retriever import Retriever
except Exception:
    class Retriever:
        def __init__(self, G=None, node_texts=None):
            self.node_texts = node_texts or {}
        def retrieve(self, question, topk_paths=5, max_hops=3, neighbor_expand=2):
            # stub result; replace with your real retrieval
            return {"paths": [["Carol","Bob"]], "contexts": ["Carol co-authored with Bob."]}

# ---------- Prompts (ASCII-safe) ----------
PROMPT_TERSE = "Answer in one short sentence using the graph evidence.\nQ: {question}\n"
PROMPT_VERBOSE = (
    "You are a helpful assistant. Use the retrieved graph paths and brief context to justify your answer.\n"
    "Question: {question}\n"
    "Reasoning paths: {paths}\n"
    "Context notes: {contexts}\n"
    "Answer:"
)

# ---------- KG loader + Reranker ----------
from .kg_loader import load_graph_from_tsv, load_node_texts
try:
    from .reranker import rerank_paths
except Exception:
    def rerank_paths(question, paths, node_texts, strategy="semantic", topk_paths=5):
        return paths[:topk_paths]

# ---------- LLM helpers (robust fallbacks) ----------
def _llm_base():
    # If you use LiteLLM proxy, set LITELLM_BASE=http://localhost:11435
    # If you call Ollama directly, it usually runs at http://localhost:11434
    return os.getenv("LITELLM_BASE", "http://localhost:11434").rstrip("/")

def _llm_model():
    # For LiteLLM + Ollama use "ollama/llama3.2:3b"
    # For direct Ollama native, model name is usually "llama3.2:3b"
    return os.getenv("LLM_MODEL", "llama3.2:3b")

def llm_complete(prompt: str) -> str:
    base = _llm_base()
    model = _llm_model()

    # Try OpenAI-compatible first: /v1/chat/completions
    try:
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e_openai:
        # Fallback to native Ollama /api/chat
        try:
            native_model = model.replace("ollama/", "")
            r = requests.post(
                f"{base}/api/chat",
                json={
                    "model": native_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            return str(data)
        except Exception as e_native:
            raise RuntimeError(f"LLM call failed. OpenAI-like error: {e_openai}; Native error: {e_native}")

# Optional: skip LLM for debugging
SKIP_LLM = os.getenv("SKIP_LLM", "0") == "1"

# ---------- FastAPI ----------
app = FastAPI(title="Graph LLM-KG API")

EDGES_PATH = os.environ.get("KG_EDGES", "./data/edges.tsv")
TEXTS_PATH = os.environ.get("KG_TEXTS", "./data/node_texts.jsonl")

def _init_retriever() -> Retriever:
    try:
        G = load_graph_from_tsv(EDGES_PATH)
        print(f"[KG] Loaded graph from {EDGES_PATH}")
    except Exception as e:
        print("[KG] TSV load failed, using empty graph:", e)
        import networkx as nx
        G = nx.DiGraph()
    try:
        texts = load_node_texts(TEXTS_PATH)
        print(f"[KG] Loaded node texts from {TEXTS_PATH} ({len(texts)} entries)")
    except Exception as e:
        print("[KG] Texts load failed:", e)
        texts = {}
    return Retriever(G, texts)

retriever = _init_retriever()

class AskBody(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 2
    rerank: str = "semantic"  # semantic | bm25 | off
    verbosity: str = "terse"  # terse | verbose

@app.get("/health")
def health():
    return {"ok": True, "service": "graph-llm-kg"}

@app.get("/routes")
def list_routes():
    return sorted([r.path for r in app.router.routes])

@app.post("/reload")
def reload_kg(edges_path: Optional[str] = None, texts_path: Optional[str] = None):
    global EDGES_PATH, TEXTS_PATH, retriever
    if edges_path:
        os.environ["KG_EDGES"] = edges_path
        EDGES_PATH = edges_path
    if texts_path:
        os.environ["KG_TEXTS"] = texts_path
        TEXTS_PATH = texts_path
    retriever = _init_retriever()
    return {"status": "ok", "edges": EDGES_PATH, "texts": TEXTS_PATH}

@app.post("/ask")
def ask(body: AskBody):
    res: Dict[str, Any] = retriever.retrieve(
        body.question,
        body.topk_paths,
        body.max_hops,
        body.neighbor_expand
    )
    paths = res.get("paths", [])
    contexts = res.get("contexts", [])

    # Rerank (safe no-op if stub)
    try:
        res["paths"] = rerank_paths(
            question=body.question,
            paths=paths,
            node_texts=getattr(retriever, "node_texts", {}) or {},
            strategy=body.rerank,
            topk_paths=body.topk_paths,
        )
    except Exception as e:
        print("Rerank error:", e)

    # Prompt
    if body.verbosity == "verbose":
        prompt = PROMPT_VERBOSE.format(
            question=body.question,
            paths=json.dumps(res.get("paths", []), ensure_ascii=False),
            contexts=json.dumps(contexts, ensure_ascii=False),
        )
    else:
        prompt = PROMPT_TERSE.format(question=body.question)

    if SKIP_LLM:
        return {"answer": "[LLM skipped]", "ctx": res, "debug": {"prompt": prompt}}

    try:
        answer = llm_complete(prompt)
        return {"answer": answer, "ctx": res}
    except Exception as e:
        return {
            "answer": "",
            "error": f"LLM error: {e}",
            "ctx": res,
            "hint": "Check LITELLM_BASE/LLM_MODEL and that your proxy or Ollama API is reachable."
        }
