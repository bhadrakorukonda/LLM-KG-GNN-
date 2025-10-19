from __future__ import annotations
import os, requests

class LLMError(RuntimeError): pass

def generate(prompt: str, *, model: str|None=None, temperature: float=0.2, max_tokens: int=256, timeout_s: float=60.0) -> str:
    """
    Default backend: Ollama HTTP /api/generate
    Configure via env:
      OLLAMA_BASE_URL (default http://127.0.0.1:11434)
      OLLAMA_MODEL    (default llama3)
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL", "llama3")
    url = f"{base}/api/generate"
    try:
        r = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "options": {"temperature": float(temperature)},
                "stream": False,
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        out = (data.get("response") or "").strip()
        if not out:
            raise LLMError("Empty response from LLM")
        return out
    except requests.RequestException as e:
        raise LLMError(f"LLM HTTP error: {e}")
