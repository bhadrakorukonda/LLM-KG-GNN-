import json, requests, streamlit as st

st.set_page_config(page_title="Graph LLM–KG v1", layout="wide")
st.title("Graph LLM–KG v1")

with st.sidebar:
    st.markdown("### API")
    api_base = st.text_input("Base URL", "http://localhost:8001")
    use_llm = st.checkbox("Use LLM (/ask_llm)", value=False)
    rel_filter = st.text_input("Relation whitelist (comma-separated)", value="")

q = st.text_input("Ask a question", value="Who co-authored a paper with Carol?")
c1, c2, c3, c4 = st.columns(4)
with c1: topk = st.number_input("Top-K paths", 1, 20, 5)
with c2: max_hops = st.number_input("Max hops", 1, 6, 3)
with c3: neighbor_expand = st.number_input("Neighborhood hops", 1, 6, 2)
with c4: dry_run = st.checkbox("Dry run", value=True, disabled=use_llm)

if st.button("Ask"):
    try:
        payload = {
            "question": q,
            "topk_paths": int(topk),
            "max_hops": int(max_hops),
            "neighbor_expand": int(neighbor_expand),
        }
        r = [r.strip() for r in rel_filter.split(",") if r.strip()]
        if r:
            payload["relation_whitelist"] = r

        path = "/ask_llm" if use_llm else f"/ask?dry_run={'true' if dry_run else 'false'}"
        url = f"{api_base}{path}"
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        st.subheader("Answer")
        st.write(data.get("answer",""))

        ctx = data.get("ctx", {})
        with st.expander("Reasoning paths", expanded=True):
            paths = ctx.get("paths", [])
            if not paths:
                st.write("_No paths returned._")
            else:
                for i, p in enumerate(paths, 1):
                    triples = p.get("triples", [])
                    pretty = " → ".join([f"{t['src']} —[{t['rel']}]→ {t['dst']}" for t in triples])
                    st.markdown(f"**{i}.** {pretty}")
        with st.expander("Raw ctx"):
            st.json(ctx)
    except Exception as e:
        st.error(f"Request failed: {e}")
