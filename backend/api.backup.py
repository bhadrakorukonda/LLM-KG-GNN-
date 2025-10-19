from app.backend.retriever import load_graph, retrieve
from app.backend.composer import build_messages
from app.backend.llm_client import chat
from app.backend.brain import answer_from_graph


load_dotenv()
load_graph()

app = FastAPI(title="Graph LLM–KG API")

class AskBody(BaseModel):
    question: str
    topk_paths: int = int(os.getenv("TOPK_PATHS", "5"))
    max_hops: int = int(os.getenv("MAX_HOPS", "3"))
    neighbor_expand: int = int(os.getenv("NEIGHBOR_EXPAND", "2"))

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
async def ask(body: AskBody, dry_run: bool = Query(default=False)):
    try:
        ctx = retrieve(body.question, body.topk_paths, body.max_hops, body.neighbor_expand)
        if dry_run:
            return {"answer": None, "ctx": ctx, "note": "dry_run (LLM skipped)"}

        # 1) Try graph-only rule for instant answers
        rule_ans = answer_from_graph(body.question, ctx)
        if rule_ans:
            return {"answer": rule_ans, "ctx": ctx, "source": "graph-rule"}

        # 2) Fall back to LLM for phrasing
        msgs = build_messages(body.question, ctx)
        t0 = time.time()
        ans = await chat(msgs)
        t1 = time.time()
        return {"answer": ans, "ctx": ctx, "latency_s": round(t1 - t0, 3), "source": "llm"}
    except Exception as e:
        return {"error": str(e)}
