from src.rag.pipeline import Pipeline

def test_ask_runs():
    p = Pipeline()
    res = p.ask("Who focuses on EGAT?")
    assert isinstance(res.answer, str)
    assert len(res.paths) > 0
