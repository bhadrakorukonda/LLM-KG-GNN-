import json
import networkx as nx
from pathlib import Path
from typing import Dict, List

class KGLite:
    def __init__(self, edge_path: str, node_texts_path: str):
        self.graph = nx.DiGraph()
        self.texts: Dict[str, str] = {}
        self._load(edge_path, node_texts_path)

    def _load(self, edge_path: str, node_texts_path: str):
        for line in Path(node_texts_path).read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            self.texts[obj["id"]] = obj["text"]
            self.graph.add_node(obj["id"])
        for line in Path(edge_path).read_text(encoding="utf-8").splitlines():
            s, r, t = line.split("\t")
            self.graph.add_edge(s, t, rel=r)

    def neighbors(self, node: str):
        return list(self.graph.successors(node)) + list(self.graph.predecessors(node))

    def subgraph_around(self, seeds: List[str], hops: int = 2):
        nodes, frontier = set(seeds), set(seeds)
        for _ in range(hops):
            nxt = set()
            for u in list(frontier):
                nxt.update(self.neighbors(u))
            frontier = nxt - nodes
            nodes.update(frontier)
        return self.graph.subgraph(nodes).copy()

    def fact_dump(self, sg: nx.Graph) -> List[str]:
        facts = []
        for u, v, d in sg.edges(data=True):
            rel = d.get("rel", "related_to")
            facts.append(f"({u}) -[{rel}]-> ({v})")
        return facts
