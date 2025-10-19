import os
API_URL = os.getenv("API_URL","http://localhost:8001")
import json, requests, streamlit as st

st.set_page_config(page_title="Graph RAG Sprint", layout="wide")
st.title("Graph RAG Sprint")

with st.sidebar:
    st.markdown("### API")
    api_base = st.text_input("Base URL", "http://localhost:8001")
    st.markdown("### Retrieval Settings")
    topk = st.number_input("Top-K paths", 1, 50, 5)
    max_hops = st.number_input("Max hops", 1, 10, 3)
    neighbor_expand = st.number_input("Neighborhood hops", 0, 5, 1)
    use_rerank = st.checkbox("Use reranker", value=True)
    rerank_mode = st.selectbox("Rerank mode", ["hybrid","bm25","fuzz"], index=0)
    # fetch available models from backend
    try:
        with st.spinner("Fetching models..."):
            r = requests.get(f"{api_base}/models", timeout=5)
            if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                models = r.json()
            else:
                models = ["ollama/llama3", "ollama/llama3.2:3B"]
    except Exception:
        models = ["ollama/llama3", "ollama/llama3.2:3B"]
    model = st.selectbox("Model", models, index=0)
    dry_run = st.checkbox("Dry run (skip LLM)", value=False)
    path_limit = st.selectbox("Paths to display", options=[10,20,50,"All"], index=0)

    st.markdown("### Data Reload")
    edges_path = st.text_input("Edges TSV", "./data/edges.tsv")
    node_texts_path = st.text_input("Node texts JSONL", "./data/node_texts.jsonl")
    if st.button("Reload Graph"):
        try:
            resp = requests.post(f"{api_base}/reload", json={"edges": edges_path, "node_texts": node_texts_path}, timeout=60)
            if resp.status_code != 200:
                st.error(f"Reload failed: {resp.text}")
            else:
                st.success(f"Reloaded: {resp.json()}")
        except Exception as e:
            st.error(f"Reload request failed: {e}")

q = st.text_input("Ask a question", value="Who co-authored a paper with Carol?")

if st.button("Ask"):
    try:
        payload = {
            "question": q,
            "topk_paths": int(topk),
            "max_hops": int(max_hops),
            "neighbor_expand": int(neighbor_expand),
            "use_rerank": bool(use_rerank),
            "rerank_mode": rerank_mode,
            "model": model,
        }
        url = f"{api_base}/ask?dry_run={'true' if dry_run else 'false'}"
        with st.spinner("Asking backend..."):
            resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("error"):
            st.warning(data.get("error"))
        else:
            if not dry_run:
                st.subheader("Answer")
                st.write(data.get("answer", ""))

            ctx = data.get("ctx", {})
            with st.expander("Paths", expanded=True):
                paths = ctx.get("paths", [])
                if not paths:
                    st.write("_No paths returned._")
                else:
                    shown = paths if path_limit == "All" else paths[:int(path_limit)]
                    for i, p in enumerate(shown, 1):
                        # p may be list of (h, r, t) tuples
                        try:
                            pretty = " → ".join([f"{h} —[{r}]→ {t}" for (h, r, t) in p])
                        except Exception:
                            pretty = str(p)
                        st.markdown(f"**{i}.** {pretty}")
            with st.expander("Context (seeds, node notes, raw)"):
                st.markdown("**Seeds**")
                st.write(ctx.get("seeds", []))
                st.markdown("**Node notes**")
                st.write(ctx.get("node_notes", []))
                st.markdown("**Raw ctx**")
                st.json(ctx)
            with st.expander("Copy request"):
                qs = f"dry_run={'true' if dry_run else 'false'}"
                st.code(json.dumps(payload, indent=2))
                st.code(f"POST /ask?{qs}")
    except Exception as e:
        st.error(f"Request failed: {e}")

