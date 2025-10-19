from fastapi import FastAPI
from backend.api.routes import router as api_router

app = FastAPI(title="Graph RAG Sprint API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"name": "Graph RAG Sprint API", "version": "0.1.0"}

# mount API routes (reload/ask)
app.include_router(api_router)
