import streamlit as st
import requests
import os
from typing import List
from src.rag.pipeline import Pipeline  # fallback local pipeline

st.set_page_config(page_title="Graph LLM-KG Demo", layout="wide")

st.title("🔗 Graph LLM–KG Demo")
st.caption("Edge-aware path retrieval + concise LLM answers. API optional.")

API_URL = os.getenv("API_URL","http://localhost:8000")
USE_API = st.toggle("Use FastAPI backend", value=False)

q = st.text_input("Ask a question", value="Who co-authored a paper with Carol?")

colA, colB = st.columns([2,3])

def normalize_paths(obj) -> List[List[str]]:
    if isinstance(obj, list):
        out = []
        for p in obj:
            if isinstance(p, list):
                out.append([str(x) for x in p])
            else:
                out.append([str(p)])
        return out
    return [[str(obj)]]

if st.button("Ask") and q.strip():
    data = None
    try:
        if USE_API:
            r = requests.post(f"{API_URL}/ask", json={"question": q}, timeout=30)
            r.raise_for_status()
            data = r.json()
        else:
            pipe = Pipeline()
            res = pipe.ask(q)
            data = {"answer": res.answer, "paths": res.paths, "context": res.context, "model": res.model}
    except Exception as e:
        st.error(f"Error while answering: {e}")

    if data:
        answer = str(data.get("answer",""))
        paths = normalize_paths(data.get("paths", []))
        model = str(data.get("model",""))
        context = str(data.get("context",""))

        with colA:
            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Model: {model}")

        with colB:
            st.subheader("Reasoning paths")
            if paths:
                for i, p in enumerate(paths, 1):
                    try:
                        st.write(f"{i}. " + " → ".join(p))
                    except Exception:
                        st.write(f"{i}. {p}")
            else:
                st.write("(no paths)")

        with st.expander("Show retrieved context"):
            st.code(context or "(empty)", language="text")
