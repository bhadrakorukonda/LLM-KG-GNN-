# src/brain.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- tiny helpers -------------------------------------------------------------

def _lower(s: Optional[str]) -> str:
    return (s or "").lower()

def _question_subject(q: str, candidates: Iterable[str]) -> Optional[str]:
    ql = _lower(q)
    for c in candidates:
        if c and _lower(c) in ql:
            return c
    return None

def _iter_triples(ctx: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    if not ctx:
        return []
    for p in ctx.get("paths") or []:
        for t in p.get("triples") or []:
            if isinstance(t, dict):
                yield t

# --- core formatter -----------------------------------------------------------

def answer_from_graph(question: str, ctx: Dict[str, Any]) -> Optional[str]:
    """
    Compose a short, deterministic sentence directly from graph triples in ctx.
    Relations handled:
      - coauthored_with (symmetric)  -> "A co-authored a paper with B."
      - collaborates_with (symmetric)-> "A collaborates with B."
      - advised_by (directional)     -> "Advisor advised Student."   (edge is Student --advised_by--> Advisor)
    Returns None if no suitable triple is found, so the caller can fall back to LLM.
    """
    triples = list(_iter_triples(ctx))
    if not triples:
        return None

    # Collect nodes mentioned in triples; useful to detect the subject mentioned in the question.
    nodes = set()
    for t in triples:
        nodes.add(t.get("src"))
        nodes.add(t.get("dst"))
    nodes.discard(None)
    subject = _question_subject(question or "", nodes)

    # Prefer exact single-edge answers that match the question subject (when present).
    # We also allow multiple results and will format the most relevant one.
    # 1) coauthored_with
    coauth = [t for t in triples if t.get("rel") == "coauthored_with" and t.get("src") and t.get("dst")]
    if coauth:
        # If multiple, try to pick the one touching the subject
        if subject:
            for t in coauth:
                if subject in (t["src"], t["dst"]):
                    s, o = (t["src"], t["dst"])
                    # If the question mentions Carol, show: "Bob co-authored a paper with Carol."
                    # Choose phrasing that names the *other* party first if that reads more natural for "Who ..." questions
                    if subject == s:
                        return f"{o} co-authored a paper with {s}."
                    else:
                        return f"{s} co-authored a paper with {o}."
        # Fallback: stable deterministic order from the first triple
        t = coauth[0]
        return f"{t['dst']} co-authored a paper with {t['src']}."

    # 2) collaborates_with
    collab = [t for t in triples if t.get("rel") == "collaborates_with" and t.get("src") and t.get("dst")]
    if collab:
        if subject:
            for t in collab:
                if subject in (t["src"], t["dst"]):
                    s, o = (t["src"], t["dst"])
                    if subject == s:
                        return f"{s} collaborates with {o}."
                    else:
                        return f"{o} collaborates with {s}."
        t = collab[0]
        return f"{t['src']} collaborates with {t['dst']}."

    # 3) advised_by (Student --advised_by--> Advisor) → "Advisor advised Student."
    advised = [t for t in triples if t.get("rel") == "advised_by" and t.get("src") and t.get("dst")]
    if advised:
        # Prefer a triple touching the subject mentioned in the question
        if subject:
            for t in advised:
                if subject in (t["src"], t["dst"]):
                    student, advisor = t["src"], t["dst"]
                    return f"{advisor} advised {student}."
        # Fallback to the first one
        student, advisor = advised[0]["src"], advised[0]["dst"]
        return f"{advisor} advised {student}."

    # Nothing we know how to template
    return None
