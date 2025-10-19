from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os, re, json, csv, time, io
import networkx as nx
from dotenv import load_dotenv

# ----------------
# Config
# ----------------
load_dotenv()
APP_NAME = os.getenv("APP_NAME", "Graph RAG Sprint API")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))
EDGES_PATH = os.path.join(DATA_DIR, os.getenv("EDGES_FILE", "edges.tsv"))
TEXTS_PATH = os.path.join(DATA_DIR, os.getenv("NODE_TEXTS_FILE", "node_texts.jsonl"))
ALIASES_PATH = os.path.join(DATA_DIR, os.getenv("ALIASES_FILE", "aliases.json"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# State
G = nx.DiGraph()
NODE_TEXT: Dict[str, str] = {}
ALIASES: Dict[str, str] = {}
STATS = {"last_reload": None, "edge_count": 0, "node_count": 0, "relations": {}}

# Models
class AskPayload(BaseModel):
    question: str
    topk_paths: int = 5
    max_hops: int = 3
    neighbor_expand: int = 2
    relation_whitelist: Optional[List[str]] = None

# ----------------
# Robust readers
# ----------------
def _read_file_text(path: str) -> str:
    # try utf-8 first, then ascii
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "ascii"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: replace errors
    return raw.decode("utf-8", errors="replace")

def _read_edges(path: str):
    text = _read_file_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows = []
    for line in lines:
        cols = re.split(r"\t+|\s{2,}", line.strip())
        if len(cols) < 3:
            continue
        if cols[0].lower() == "src" and cols[1].lower() == "rel":
            continue
        src, rel, dst = (c.strip() for c in cols[:3])
        if src and rel and dst:
            rows.append((src, rel, dst))
    return rows

def _read_texts(path: str) -> Dict[str,str]:
    txt = {}
    text = _read_file_text(path)
    for line in text.splitlines():
        line=line.strip()
        if not line: 
            continue
        obj = json.loads(line)
        i = obj.get("id","").strip()
        t = obj.get("text","").strip()
        if i: txt[i]=t
    return txt

def _read_aliases(path: str) -> Dict[str,str]:
    if not os.path.exists(path): 
        return {}
    text = _read_file_text(path)
    raw = json.loads(text or "{}")
    return {str(k).lower(): str(v) for k,v in raw.items()}

def _apply_relation_whitelist(Gin: nx.DiGraph, allow: Optional[List[str]]) -> nx.DiGraph:
    if not allow: 
        return Gin
    allow_set = set(allow)
    H = nx.DiGraph()
    H.add_nodes_from(Gin.nodes(data=True))
    for u, v, data in Gin.edges(data=True):
        if data.get("rel") in allow_set:
            H.add_edge(u, v, **data)
    return H

def _k_shortest_simple_paths(Gin: nx.DiGraph, src: str, dst: str, k: int, max_hops:int):
    if src not in Gin or dst not in Gin: 
        return []
    all_paths = []
    frontier = [(src, [src])]
    seen_paths = set()
    for _depth in range(max_hops+1):
        new_frontier = []
        for node, path in frontier:
            if node == dst and len(path) > 1:
                triples = []
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    rel = Gin[u][v].get("rel")
                    triples.append((u, rel, v))
                tup = tuple(triples)
                if tup not in seen_paths:
                    seen_paths.add(tup)
                    all_paths.append(triples)
                if len(all_paths) >= k:
                    return all_paths
            for _, v, data in Gin.out_edges(node, data=True):
                if v in path: 
                    continue
                new_frontier.append((v, path+[v]))
        frontier = new_frontier
        if not frontier:
            break
    return all_paths[:k]

def _extract_names_from_question(q: str, names: List[str]) -> List[str]:
    tokens = [t.strip(".,?!;:()[]\"'") for t in q.lower().split()]
    name_map = {n.lower(): n for n in names}
    out = [name_map[t] for t in tokens if t in name_map]
    for t in tokens:
        canon = ALIASES.get(t)
        if canon and canon in name_map.values() and canon not in out:
            out.append(canon)
    return out

def _format_paths(triples_paths):
    return [{"triples":[{"src":a,"rel":r,"dst":b} for (a,r,b) in triples],
             "length":len(triples)} for triples in triples_paths]

def _llm_answer(paths, question):
    if not paths: return "I don't know."
    try:
        from litellm import completion
        path_strs = []
        for p in paths[:3]:
            segs = [f"{t['src']} -[{t['rel']}]-> {t['dst']}" for t in p["triples"]]
            path_strs.append(" ; ".join(segs))
        context = "\n".join(f"- {s}" for s in path_strs)
        prompt = f"""Use ONLY these graph paths to answer.
Paths:
{context}

Question: {question}
Answer briefly and cite the key relation."""
        resp = completion(
            model=LLM_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=float(os.getenv("LLM_TEMPERATURE","0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS","256"))
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return "A path exists consistent with the question."

# ----------------
# Routes
# ----------------
@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.process_time())}

@app.get("/version")
def version():
    return {"name": APP_NAME, "version": APP_VERSION}

@app.get("/debug/paths")
def debug_paths():
    def stat(p):
        return {"exists": os.path.exists(p),
                "abspath": os.path.abspath(p),
                "size": (os.path.getsize(p) if os.path.exists(p) else None)}
    return {
        "cwd": os.getcwd(),
        "DATA_DIR": DATA_DIR,
        "EDGES_PATH": stat(EDGES_PATH),
        "TEXTS_PATH": stat(TEXTS_PATH),
        "ALIASES_PATH": stat(ALIASES_PATH)
    }

@app.get("/graph/stats")
def graph_stats():
    return {
        "nodes": STATS.get("node_count", 0),
        "edges": STATS.get("edge_count", 0),
        "relations": STATS.get("relations", {}),
        "last_reload": STATS.get("last_reload")
    }

@app.post("/reload")
def reload_data():
    global G, NODE_TEXT, ALIASES, STATS
    try:
        if not os.path.exists(EDGES_PATH):
            return {"status":"error","error":f"missing {EDGES_PATH}"}
        if not os.path.exists(TEXTS_PATH):
            return {"status":"error","error":f"missing {TEXTS_PATH}"}

        edges = _read_edges(EDGES_PATH)
        NODE_TEXT = _read_texts(TEXTS_PATH)
        ALIASES = _read_aliases(ALIASES_PATH)

        G = nx.DiGraph()
        for src, rel, dst in edges:
            G.add_node(src)
            G.add_node(dst)
            G.add_edge(src, dst, rel=rel)

        rel_count = defaultdict(int)
        for _, _, data in G.edges(data=True):
            rel_count[data.get("rel","__none__")] += 1
        STATS = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "relations": dict(rel_count),
            "last_reload": int(time.time())
        }

        return {
            "status": "ok",
            "edges": f"./data/{os.path.basename(EDGES_PATH)}",
            "texts": f"./data/{os.path.basename(TEXTS_PATH)}",
            "node_count": STATS["node_count"],
            "edge_count": STATS["edge_count"],
            "relations": STATS["relations"]
        }
    except Exception as e:
        # return root cause to the client to debug quickly
        return {
            "status":"error",
            "error": str(e),
            "cwd": os.getcwd(),
            "DATA_DIR": DATA_DIR,
            "EDGES_PATH": EDGES_PATH,
            "TEXTS_PATH": TEXTS_PATH,
            "ALIASES_PATH": ALIASES_PATH
        }

@app.post("/ask")
def ask(dry_run: bool = Query(default=False), payload: AskPayload = Body(...)):
    names = list(G.nodes())
    mentioned = _extract_names_from_question(payload.question, names)
    src, dst = None, None
    if len(mentioned) >= 2: src, dst = mentioned[0], mentioned[1]
    elif len(mentioned) == 1:
        src = mentioned[0]
        for _, v, data in G.out_edges(src, data=True):
            if "coauthor" in data.get("rel",""):
                dst = v; break

    Gq = _apply_relation_whitelist(G, payload.relation_whitelist)
    triples_paths = []
    if src and dst:
        triples_paths = _k_shortest_simple_paths(Gq, src, dst, k=payload.topk_paths, max_hops=payload.max_hops)

    seeds = list(G.nodes())
    node_notes = [f"{nid}: {NODE_TEXT.get(nid,'')}".strip() for nid in seeds]
    ctx = {"seeds": seeds, "paths": _format_paths(triples_paths), "contexts": [], "local_facts": [], "node_notes": node_notes}

    if dry_run:
        return {"answer": "I don't know." if not triples_paths else "Path(s) found.", "ctx": ctx, "model": f"{LLM_PROVIDER}/{LLM_MODEL}"}

    answer = _llm_answer(ctx["paths"], payload.question)
    return {"answer": answer, "ctx": ctx, "model": f"{LLM_PROVIDER}/{LLM_MODEL}"}

@app.post("/ask_llm")
def ask_llm(payload: AskPayload = Body(...)):
    names = list(G.nodes())
    mentioned = _extract_names_from_question(payload.question, names)
    src, dst = (mentioned + [None, None])[:2]
    if src and not dst:
        for _, v, data in G.out_edges(src, data=True):
            if "coauthor" in data.get("rel",""):
                dst = v; break
    Gq = _apply_relation_whitelist(G, payload.relation_whitelist)
    triples_paths = []
    if src and dst:
        triples_paths = _k_shortest_simple_paths(Gq, src, dst, k=payload.topk_paths, max_hops=payload.max_hops)
    ctx = {"seeds": list(G.nodes()), "paths": _format_paths(triples_paths), "contexts": [], "local_facts": [], "node_notes": [f"{nid}: {NODE_TEXT.get(nid,'')}".strip() for nid in G.nodes()]}
    answer = _llm_answer(ctx["paths"], payload.question)
    return {"answer": answer, "ctx": ctx, "model": f"{LLM_PROVIDER}/{LLM_MODEL}"}
