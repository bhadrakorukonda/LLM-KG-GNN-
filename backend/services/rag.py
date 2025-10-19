from __future__ import annotations
from typing import Dict, Any, List, Set
import re

from backend.services.graph import graph_store
from backend.services.paths import find_paths, neighborhood
from backend.services import llm

COAUTHOR_RELS = {"coauthored_with","coauthored","co-authored-with","co_authored_with"}

def _extract_seeds_from_question(q: str) -> List[str]:
    G = graph_store.graph()
    tokens = re.findall(r"[A-Za-z0-9_]+", q or "")
    seen = set(); seeds=[]
    for t in tokens:
        if t in seen: continue
        if t in G: seeds.append(t); seen.add(t)
    return seeds

def _collect_node_notes(nodes: Set[str]) -> List[str]:
    notes=[]
    for n in sorted(nodes):
        txt = graph_store.get_node_text(n) or ""
        notes.append(f"{n}: {txt}".rstrip())
    return notes

def _simple_answer_from_graph(question: str, seeds: List[str]) -> str:
    q_lower = (question or "").lower()
    if "co-author" in q_lower or "coauth" in q_lower:
        G = graph_store.graph()
        for s in seeds:
            for _, v, d in G.out_edges(s, data=True):
                if (d.get("rel") or "").lower() in COAUTHOR_RELS:
                    return f"{v} co-authored a paper with {s}."
            for u, _, d in G.in_edges(s, data=True):
                if (d.get("rel") or "").lower() in COAUTHOR_RELS:
                    return f"{u} co-authored a paper with {s}."
    return "I used the graph to collect reasoning paths and context."

def _build_prompt(question: str, ctx: Dict[str, Any], answer_style: str | None, explain: bool) -> str:
    lines=[]
    lines.append("You answer using ONLY the graph evidence below.")
    lines.append("If evidence is insufficient, say you don't know.")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Graph triples (reasoning paths):")
    if ctx.get("paths"):
        for i, triples in enumerate(ctx["paths"], 1):
            pretty = " -> ".join([f"{u} -[{r}]-> {v}" for (u,r,v) in triples])
            lines.append(f"{i}. {pretty}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("Node notes:")
    for n in (ctx.get("node_notes") or []):
        lines.append(f"- {n}")
    lines.append("")
    style = (answer_style or "sentence").lower()
    if style == "short":
        lines.append("Instruction: Return a single short answer (a name or phrase).")
    elif style == "bullet":
        lines.append("Instruction: Return one bullet summarizing the answer.")
    else:
        lines.append("Instruction: Return one concise sentence answering the question.")
    if explain:
        lines.append("Also include one brief justification that cites the relevant triple(s) by index (e.g., Path 1).")
    return "\n".join(lines)

def answer(req: Dict[str, Any]) -> Dict[str, Any]:
    question: str = req.get("question") or ""
    topk_paths: int = int(req.get("topk_paths") or 5)
    max_hops: int = int(req.get("max_hops") or 3)
    neighbor_expand: int = int(req.get("neighbor_expand") or 1)
    dry_run: bool = bool(req.get("dry_run") or False)

    use_llm: bool = bool(req.get("use_llm") or False)
    llm_model: str | None = req.get("llm_model")
    temperature: float = float(req.get("temperature") or 0.2)
    max_tokens: int = int(req.get("max_tokens") or 256)
    explain: bool = bool(req.get("explain") or False)
    answer_style: str | None = req.get("answer_style")

    seeds = _extract_seeds_from_question(question)
    paths = find_paths(seeds, max_hops=max_hops, topk_paths=topk_paths)

    nodes_in_paths: Set[str] = set()
    for triples in paths:
        for (u,_r,v) in triples:
            nodes_in_paths.add(u); nodes_in_paths.add(v)

    expanded_nodes = neighborhood(nodes_in_paths or seeds, hops=max(neighbor_expand,0))
    node_notes = _collect_node_notes(expanded_nodes)

    ctx = {
        "seeds": seeds,
        "paths": paths,
        "contexts": [],
        "local_facts": [],
        "node_notes": node_notes,
    }

    if dry_run:
        return {"answer": "", "ctx": ctx, "source": "graph"}

    if not use_llm:
        ans = _simple_answer_from_graph(question, seeds)
        return {"answer": ans, "ctx": ctx, "source": "graph"}

    prompt = _build_prompt(question, ctx, answer_style, explain)
    try:
        out = llm.generate(prompt, model=llm_model, temperature=temperature, max_tokens=max_tokens)
        return {"answer": out, "ctx": ctx, "source": "llm+graph"}
    except Exception as e:
        ans = _simple_answer_from_graph(question, seeds)
        return {"answer": ans, "ctx": ctx, "source": f"graph (llm_failed: {e})"}
