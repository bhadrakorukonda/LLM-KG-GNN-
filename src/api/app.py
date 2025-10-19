from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.rag.pipeline import Pipeline

app = FastAPI(title="Graph LLM-KG API", version="0.1.1")
pipe = Pipeline()

class AskIn(BaseModel):
    question: str

class AskOut(BaseModel):
    answer: str
    paths: List[List[str]]
    context: str
    model: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    res = pipe.ask(payload.question)
    return {"answer": res.answer, "paths": res.paths, "context": res.context, "model": res.model}
