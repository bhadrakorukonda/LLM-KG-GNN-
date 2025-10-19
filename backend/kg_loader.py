import csv, json
from typing import Dict
try:
    import networkx as nx
except Exception:
    nx = None

def load_graph_from_tsv(tsv_path: str):
    if nx is None:
        raise RuntimeError("networkx not available")
    G = nx.MultiDiGraph() if hasattr(nx, "MultiDiGraph") else nx.DiGraph()
    with open(tsv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            s, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if s and t:
                G.add_edge(s, t, rel=r)
    return G

def load_node_texts(jsonl_path: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            nid = str(obj.get("id") or obj.get("node") or obj.get("name") or "")
            if not nid:
                continue
            m[nid] = obj.get("text") or obj.get("desc") or ""
    return m
