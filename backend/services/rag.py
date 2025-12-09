from __future__ import annotations
from typing import Dict, Any, List, Set, Tuple
import re
import os

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

def _build_prompt_with_citations(question: str, ctx: Dict[str, Any], answer_style: str | None, explain: bool) -> str:
    """Build prompt that encourages explicit path citations"""
    lines=[]
    lines.append("You are a precise assistant that answers questions using ONLY the provided graph evidence.")
    lines.append("You MUST cite which path(s) you used by referencing their numbers in square brackets, e.g., [Path 1], [Path 2].")
    lines.append("If evidence is insufficient, say you don't know.")
    lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Graph Evidence (Retrieved Paths):")
    
    if ctx.get("paths"):
        for i, path_dict in enumerate(ctx["paths"], 1):
            # Format path text from dict
            if isinstance(path_dict, dict):
                path_text = path_dict.get("text", "")
                if not path_text and "triples" in path_dict:
                    triples = path_dict["triples"]
                    path_text = " → ".join([
                        f"{t['src']} —[{t['relation']}]→ {t['dst']}" 
                        for t in triples
                    ])
                lines.append(f"[Path {i}] {path_text}")
                
                # Include GNN score if available
                if "gnn_score" in path_dict:
                    lines.append(f"  (Confidence: {path_dict['gnn_score']:.3f})")
            else:
                # Legacy format: list of triples
                pretty = " → ".join([f"{u} —[{r}]→ {v}" for (u,r,v) in path_dict])
                lines.append(f"[Path {i}] {pretty}")
    else:
        lines.append("(no paths found)")
    
    lines.append("")
    lines.append("Additional Context:")
    for n in (ctx.get("node_notes") or []):
        lines.append(f"- {n}")
    lines.append("")
    
    style = (answer_style or "sentence").lower()
    if style == "short":
        lines.append("Instruction: Answer in one short phrase and cite the path(s) used.")
    elif style == "bullet":
        lines.append("Instruction: Answer with one bullet point and cite the path(s) used.")
    else:
        lines.append("Instruction: Answer in 1-2 sentences and cite which path(s) [Path X] you used.")
    
    if explain:
        lines.append("Also briefly explain your reasoning process step-by-step.")
    
    return "\n".join(lines)

def _extract_citations_from_answer(answer: str, paths: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract which paths were cited in the LLM answer.
    Returns list of cited paths with their indices.
    """
    cited_paths = []
    # Look for patterns like [Path 1], [Path 2], etc.
    citation_pattern = r'\[Path (\d+)\]'
    matches = re.finditer(citation_pattern, answer, re.IGNORECASE)
    
    cited_indices = set()
    for match in matches:
        idx = int(match.group(1)) - 1  # Convert to 0-based index
        if 0 <= idx < len(paths):
            cited_indices.add(idx)
    
    # Build cited paths list
    for idx in sorted(cited_indices):
        path = paths[idx]
        cited_paths.append({
            "path_index": idx + 1,
            "path": path,
            "text": path.get("text", ""),
            "confidence": path.get("gnn_score", None),
        })
    
    return cited_paths

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
    
    # Check if GNN should be used
    use_gnn = os.environ.get("USE_GNN", "false").lower() == "true"

    # Extract seed entities from question
    seeds = _extract_seeds_from_question(question)
    
    # Retrieve paths with optional GNN scoring
    paths = find_paths(seeds, max_hops=max_hops, topk_paths=topk_paths, use_gnn=use_gnn)

    nodes_in_paths: Set[str] = set()
    for path_dict in paths:
        if isinstance(path_dict, dict) and "triples" in path_dict:
            for triple in path_dict["triples"]:
                nodes_in_paths.add(triple.get("src"))
                nodes_in_paths.add(triple.get("dst"))
        else:
            # Legacy format
            for (u,_r,v) in path_dict:
                nodes_in_paths.add(u); nodes_in_paths.add(v)

    expanded_nodes = neighborhood(nodes_in_paths or seeds, hops=max(neighbor_expand,0))
    node_notes = _collect_node_notes(expanded_nodes)

    ctx = {
        "seeds": seeds,
        "paths": paths,
        "contexts": [],
        "local_facts": [],
        "node_notes": node_notes,
        "retrieval_method": "gnn" if use_gnn else "bfs",
    }

    if dry_run:
        return {"answer": "", "ctx": ctx, "source": "graph", "citations": []}

    if not use_llm:
        ans = _simple_answer_from_graph(question, seeds)
        return {"answer": ans, "ctx": ctx, "source": "graph", "citations": []}

    # Build prompt with citation instructions
    prompt = _build_prompt_with_citations(question, ctx, answer_style, explain)
    
    try:
        out = llm.generate(prompt, model=llm_model, temperature=temperature, max_tokens=max_tokens)
        
        # Extract citations from answer
        citations = _extract_citations_from_answer(out, paths)
        
        # Build reasoning trace
        reasoning_trace = {
            "step_1_entity_detection": {"seeds": seeds, "method": "regex_matching"},
            "step_2_path_retrieval": {
                "method": "gnn_ranked" if use_gnn else "bfs_search",
                "paths_found": len(paths),
                "max_hops": max_hops,
            },
            "step_3_context_expansion": {
                "neighbor_hops": neighbor_expand,
                "total_nodes": len(expanded_nodes),
            },
            "step_4_llm_generation": {
                "model": llm_model,
                "temperature": temperature,
                "paths_cited": len(citations),
            },
        }
        
        return {
            "answer": out,
            "ctx": ctx,
            "source": "llm+graph+gnn" if use_gnn else "llm+graph",
            "citations": citations,
            "reasoning_trace": reasoning_trace,
        }
    except Exception as e:
        ans = _simple_answer_from_graph(question, seeds)
        return {
            "answer": ans,
            "ctx": ctx,
            "source": f"graph (llm_failed: {e})",
            "citations": [],
        }
