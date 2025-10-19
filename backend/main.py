from fastapi import FastAPI
from backend.api import router

app = FastAPI(title="Graph RAG Sprint API", version="0.1.0")
app.include_router(router, prefix="")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"name": "Graph RAG Sprint API", "version": "0.1.0"}

# This is intentionally simple so tests can monkeypatch it
def generate_answer(question: str, ctx: dict, model: str = "ollama/llama3", timeout_s: int = 30) -> str:
    # default implementation, replaced in tests
    return "DEFAULT_ANSWER"
