import os, requests, streamlit as st

API = f"http://localhost:{os.getenv('API_PORT','8000')}"
st.set_page_config(page_title="Graph LLM–KG", layout="wide")

st.title("🔗 Graph LLM–KG Demo")
q = st.text_input("Ask a question about the graph", value="Who co-authored a paper with Carol?")

c1, c2, c3, c4 = st.columns(4)
with c1:
    topk_paths = st.number_input("Top-K paths", min_value=1, max_value=20, value=5)
with c2:
    max_hops = st.number_input("Max hops", min_value=1, max_value=6, value=3)
with c3:
    neigh = st.number_input("Neighborhood hops", min_value=1, max_value=4, value=2)
with c4:
    dry_run = st.checkbox("Dry run (skip LLM)", value=False)

if st.button("Ask"):
    with st.spinner("Thinking..."):
        params = {"dry_run": str(dry_run).lower()}
        r = requests.post(f"{API}/ask", params=params, json={
            "question": q,
            "topk_paths": int(topk_paths),
            "max_hops": int(max_hops),
            "neighbor_expand": int(neigh),
        })
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            st.error(f"Backend error: {data['error']}")
        else:
            st.subheader("Answer")
            src = data.get("source", data.get("note", ""))
            if src:
                st.caption(f"Source: {src}")
            st.write(data.get("answer", "(no answer)"))

        with st.expander("Evidence (ctx)"):
            st.json(data.get("ctx", {}))
