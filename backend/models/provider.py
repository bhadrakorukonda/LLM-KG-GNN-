from __future__ import annotations
import os
from typing import Dict, Any, List

import json
import httpx


def _summarize_without_llm(question: str, ctx: Dict[str, Any]) -> str:
    seeds = ctx.get("seeds") or []
    paths = ctx.get("paths") or []
    if not paths:
        return "I don't know."
    try:
        first = paths[0]
        if first and isinstance(first, list) and len(first) > 0:
            h, r, t = first[0]
            return f"I found a connection between {h} and {t}."
    except Exception:
        pass
    return "I don't know."


def _truncate_ctx(ctx: Dict[str, Any], max_paths: int = 8, max_notes: int = 20) -> Dict[str, Any]:
    out = {
        "seeds": list(ctx.get("seeds") or [])[:10],
        "paths": [list(p)[:10] for p in list(ctx.get("paths") or [])[:max_paths]],
        "node_notes": list(ctx.get("node_notes") or [])[:max_notes],
        "contexts": {},  # drop heavy neighborhood in prompt
    }
    return out


def generate_answer(
    question: str,
    ctx: Dict[str, Any],
    model: str = "ollama/llama3",
    timeout_s: int = 30,
    max_tokens: int = 256,
) -> str:
    """
    Generate a short answer from graph context using a pluggable provider.
    If OLLAMA_HOST is set, calls local Ollama; otherwise, returns a fallback summary.
    """
    ollama_host = os.getenv("OLLAMA_HOST")
    safe_ctx = _truncate_ctx(ctx)

    if not ollama_host:
        return _summarize_without_llm(question, safe_ctx)

    # --- Ollama local inference ---
    base = ollama_host.rstrip("/")
    # Allow model name like "ollama/llama3" or raw "llama3"
    model_name = model.split("/", 1)[-1] if "/" in model else model
    # Build compact prompt: question + top N triples + notes
    triples: List[str] = []
    for p in safe_ctx.get("paths", []) or []:
        try:
            for (h, r, t) in p:
                triples.append(f"{h} -[{r}]-> {t}")
        except Exception:
            continue
        if len(triples) >= 50:
            break
    notes = safe_ctx.get("node_notes", []) or []
    prompt = (
        "Use only the graph facts to answer.\n"
        f"Question: {question}\n"
        "Facts:\n- " + "\n- ".join(triples[:50]) + "\n"
        "Notes:\n- " + "\n- ".join(notes[:20]) + "\n"
        "Answer in 1-2 sentences."
    )

    try:
        timeout = httpx.Timeout(connect=3.0, read=45.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            if resp.status_code != 200:
                return _summarize_without_llm(question, safe_ctx)
            data = resp.json()
            return (data.get("response") or "").strip() or _summarize_without_llm(question, safe_ctx)
    except Exception:
        # warn fallback
        import logging
        logging.getLogger("graph-rag").warning("ollama_fallback")
        return _summarize_without_llm(question, safe_ctx)


