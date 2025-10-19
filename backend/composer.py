SYSTEM = """You are a graph-grounded assistant.
Use only the graph paths and local facts as evidence.
If unsure, say you don't know. Always list the key evidence."""

USER_TEMPLATE = """Question: {question}

Top reasoning paths:
{paths}

Local facts:
{facts}

Node notes (summaries):
{notes}

Instructions:
1) Answer concisely (2-5 sentences).
2) Cite the minimal set of paths/facts as 'Evidence'.
3) If the graph doesn't contain enough info, say so and suggest 1-2 follow-ups."""

def build_messages(q: str, ctx: dict) -> list[dict]:
    paths = "\n".join(f"- {c}" for c in ctx.get("contexts", []))
    facts = "\n".join(f"- {c}" for c in ctx.get("local_facts", []))
    notes = "\n".join(f"- {c}" for c in ctx.get("node_notes", []))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=q,
            paths=paths or "(none)",
            facts=facts or "(none)",
            notes=notes or "(none)",
        )},
    ]
