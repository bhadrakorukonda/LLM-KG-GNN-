import re
import networkx as nx
from typing import List
from src.kg.loaders import KGLite

ENTITY_RE = re.compile(r"\b(alice|bob|carol|dave|eve)\b", re.I)

class SimplePathRetriever:
    def __init__(self, kg: KGLite):
        self.kg = kg

    def seed_entities(self, question: str) -> List[str]:
        seeds = list({m.group(1)[0].upper() for m in ENTITY_RE.finditer(question or "")})
        return seeds or list(self.kg.graph.nodes())[:1]

    def top_paths(self, question: str, k: int = 3, max_len: int = 3) -> List[List[str]]:
        seeds = self.seed_entities(question)
        paths: List[List[str]] = []
        G = self.kg.graph
        for s in seeds:
            for t in G.nodes():
                if s == t:
                    continue
                try:
                    sp = list(map(str, nx.shortest_path(G.to_undirected(), s, t)))
                    if 1 <= len(sp)-1 <= max_len:
                        paths.append(sp)
                except nx.NetworkXNoPath:
                    pass
        def score(p):
            ent_bonus = 1 if ENTITY_RE.search(question or "") else 0
            return (-len(p), ent_bonus)
        paths.sort(key=score)
        uniq, seen = [], set()
        for p in paths:
            key = tuple(p)
            if key not in seen:
                uniq.append(p); seen.add(key)
            if len(uniq) >= k:
                break
        return uniq

    def context(self, paths: List[List[str]]) -> str:
        sg_nodes = set([n for p in paths for n in p])
        sg = self.kg.graph.subgraph(sg_nodes).copy()
        facts = self.kg.fact_dump(sg)
        node_notes = [f"{n}: {self.kg.texts.get(n,'')}" for n in sg_nodes]
        return "Facts:\n" + "\n".join(facts) + "\n\nNotes:\n" + "\n".join(node_notes)
