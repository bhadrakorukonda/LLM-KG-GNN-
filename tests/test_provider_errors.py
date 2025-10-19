import httpx
from backend.models import provider


def test_provider_fallback_on_connect_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    def fake_post(self, url, json=None):
        raise httpx.ConnectError("boom", request=None)

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    out = provider.generate_answer(
        question="Who co-authored with Carol?",
        ctx={"paths": [[("Alice","coauthored_with","Carol")]], "seeds": ["Carol"], "node_notes": []},
        model="ollama/llama3",
    )
    assert isinstance(out, str) and out != ""

