from dataclasses import dataclass
from typing import List
from pathlib import Path
from src.kg.loaders import KGLite
from src.rag.retrieval import SimplePathRetriever
from src.llm.openai_client import llm_answer

@dataclass
class AskResult:
    answer: str
    paths: List[List[str]]
    context: str
    model: str

class Pipeline:
    def __init__(self, data_dir: str = "src/data/toy"):
        edge = Path(data_dir)/"kg_edges.tsv"
        txts = Path(data_dir)/"node_texts.jsonl"
        self.kg = KGLite(str(edge), str(txts))
        self.retriever = SimplePathRetriever(self.kg)

    def ask(self, question: str, topk_paths: int = 3) -> AskResult:
        paths = self.retriever.top_paths(question, k=topk_paths)
        ctx = self.retriever.context(paths)
        out = llm_answer(question, ctx)
        return AskResult(answer=out["answer"], paths=paths, context=ctx, model=out["model"])
