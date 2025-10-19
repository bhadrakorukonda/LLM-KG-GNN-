import os, httpx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
MODEL = os.getenv("LLM_MODEL", "ollama/llama3")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMP = float(os.getenv("TEMPERATURE", "0.2"))

_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=20.0, pool=5.0)

async def chat(messages: List[Dict[str, str]]) -> str:
    url = f"{BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
