import os
from typing import Dict
import requests

# Routing mode: "auto" (local then cloud), "local", or "openai"
ROUTING = os.getenv("LLM_ROUTING", "auto").lower()

# Default logical model name (used by both)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM = "You answer concisely and cite node IDs like [A], [B]."

# OpenAI (cloud)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# Local OpenAI-compatible server (LiteLLM proxy → Ollama)
LOCAL_API_BASE  = os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1")
LOCAL_MODEL     = os.getenv("LOCAL_MODEL", "ollama/llama3")
LOCAL_API_KEY   = os.getenv("LOCAL_API_KEY", "")  # usually not needed

BAD_MARKERS = ("i cannot","as an ai","do not have access","unable to","i'm sorry")

def _post(base_url: str, api_key: str, payload: dict, timeout: float = 12.0) -> Dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_cloud(question: str, context: str, timeout: float = 12.0) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for cloud route")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer with a name and cite nodes in square brackets."},
        ],
        "temperature": 0.2,
    }
    data = _post(OPENAI_API_BASE, OPENAI_API_KEY, payload, timeout=timeout)
    txt = data["choices"][0]["message"]["content"]
    return {"answer": txt, "model": MODEL, "source": "cloud"}

def _call_local(question: str, context: str, timeout: float = 6.0) -> Dict[str, str]:
    payload = {
        "model": os.getenv("OPENAI_MODEL", LOCAL_MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer with a name and cite nodes in square brackets."},
        ],
        "temperature": 0.2,
    }
    data = _post(LOCAL_API_BASE, LOCAL_API_KEY, payload, timeout=timeout)
    txt = data["choices"][0]["message"]["content"]
    return {"answer": txt, "model": payload["model"], "source": "local"}

def _stub_from_context(context: str) -> Dict[str, str]:
    guess = "Unknown"
    for line in context.splitlines():
        if "->" in line:
            lhs = line.split("->")[0]
            cand = lhs.split("(")[-1].split(")")[0]
            if cand:
                guess = cand; break
    return {"answer": f"{guess} [demo-mode]", "model": "stub", "source": "stub"}

def _is_good_enough(answer: str, context: str) -> bool:
    if not answer or len(answer.strip()) < 4: return False
    low = answer.lower()
    if any(b in low for b in BAD_MARKERS): return False
    if "[" in answer and "]" in answer: return True
    for nid in ("A","B","C","D","E"):
        if nid in context and nid in answer: return True
    return len(answer.strip()) >= 12

def llm_answer(question: str, context: str) -> Dict[str, str]:
    mode = ROUTING
    if mode == "openai":
        try: return _call_cloud(question, context)
        except Exception: return _stub_from_context(context)
    if mode == "local":
        try: return _call_local(question, context)
        except Exception: return _stub_from_context(context)

    # auto: local → (if bad/err) cloud → (if err) stub
    try:
        local = _call_local(question, context, timeout=6.0)
        if _is_good_enough(local["answer"], context): return local
    except Exception:
        pass
    try:
        cloud = _call_cloud(question, context, timeout=12.0)
        return cloud
    except Exception:
        return _stub_from_context(context)
