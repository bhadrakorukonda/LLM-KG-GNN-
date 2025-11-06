# backend/llm.py
from __future__ import annotations
import os, requests, json
from typing import List

def _ollama_base() -> str:
    return os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

def list_models():
    r=requests.get(f"{_ollama_base()}/api/tags", timeout=10)
    r.raise_for_status()
    data=r.json()
    return [m["name"] for m in data.get("models", [])]

def generate_answer(model: str, question: str, paths: List[List[str]]) -> str:
    """
    Compose a concise answer grounded in provided graph paths.
    """
    prompt = (
        "You are a concise graph QA assistant.\n"
        "Use the reasoning paths as evidence. If unknown, say you don't know.\n\n"
        f"Question: {question}\n"
        f"Reasoning paths (node chains):\n" +
        "\n".join(f"- {' → '.join(p)}" for p in paths[:10]) +
        "\n\nAnswer briefly with names and relationships inferred."
    )
    payload={"model": model, "prompt": prompt, "stream": False}
    r=requests.post(f"{_ollama_base()}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    out=r.json()
    return out.get("response","").strip()
