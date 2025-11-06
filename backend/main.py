# backend/main.py
from __future__ import annotations

"""
Compatibility module for tests.

- tests import backend.main.app (we proxy the FastAPI app from backend.api)
- tests monkeypatch backend.main.generate_answer (so expose a function here)
"""

from backend.api import app  # re-export the running FastAPI app


def generate_answer(
    question: str,
    ctx: dict,
    model: str = "ollama/llama3",
    timeout_s: int = 30,
) -> str:
    """
    Default implementation; tests will monkeypatch this.
    Keep it simple and deterministic.
    """
    return "DEFAULT_ANSWER"


__all__ = ["app", "generate_answer"]
