import os
import re
import json
import requests
import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Any

API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(page_title="GRAIL-LM: Graph-RAG with Neo4j & GNN", layout="wide")

# Header with logo/branding
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 GRAIL-LM")
    st.caption("Graph-RAG with Neo4j, PyTorch Geometric GNN, and LLaMA3")
with col2:
    st.metric("Status", "🟢 Online")

# ---------- Visualization Functions ----------
def visualize_reasoning_trace(trace: Dict[str, Any]):
    """Visualize step-by-step reasoning process"""
    if not trace:
        return
    
    st.markdown("### 🔍 Reasoning Pipeline")
    
    steps = []
    if "step_1_entity_detection" in trace:
        seeds = trace["step_1_entity_detection"].get("seeds", [])
        steps.append(f"**1. Entity Detection**: {len(seeds)} entities found: {', '.join(seeds)}")
    
    if "step_2_path_retrieval" in trace:
        ret = trace["step_2_path_retrieval"]
        method = ret.get("method", "unknown")
        method_label = "🧮 GNN-Ranked" if "gnn" in method else "🔍 BFS Search"
        steps.append(f"**2. Path Retrieval**: {method_label} → {ret.get('paths_found', 0)} paths (max {ret.get('max_hops', 3)} hops)")
    
    if "step_3_context_expansion" in trace:
        exp = trace["step_3_context_expansion"]
        steps.append(f"**3. Context Expansion**: {exp.get('neighbor_hops', 0)} hops → {exp.get('total_nodes', 0)} total nodes")
    
    if "step_4_llm_generation" in trace:
        llm_info = trace["step_4_llm_generation"]
        steps.append(f"**4. Answer Generation**: {llm_info.get('model', 'N/A')} (T={llm_info.get('temperature', 0.2)}) → {llm_info.get('paths_cited', 0)} paths cited")
    
    for step in steps:
        st.markdown(step)


def visualize_path_graph(paths: List[Dict[str, Any]], citations: List[Dict] = None):
    """
    Create interactive graph visualization of retrieved paths using Plotly.
    Highlights cited paths.
    """
    if not paths:
        st.info("No paths to visualize")
        return
    
    # Extract all nodes and edges
    nodes_set = set()
    edges_list = []
    cited_indices = set(c["path_index"] - 1 for c in (citations or []))
    
    for idx, path in enumerate(paths):
        is_cited = idx in cited_indices
        
        if isinstance(path, dict) and "triples" in path:
            for triple in path["triples"]:
                src = triple.get("src", "")
                dst = triple.get("dst", "")
                rel = triple.get("relation", "")
                nodes_set.add(src)
                nodes_set.add(dst)
                
                # Mark edge properties
                edges_list.append({
                    "source": src,
                    "target": dst,
                    "relation": rel,
                    "cited": is_cited,
                    "path_idx": idx + 1,
                    "gnn_score": path.get("gnn_score")
                })
    
    if not nodes_set:
        st.info("No graph structure found in paths")
        return
    
    # Position nodes in a circular layout
    import math
    nodes = list(nodes_set)
    n = len(nodes)
    node_positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        node_positions[node] = (math.cos(angle), math.sin(angle))
    
    # Create Plotly figure
    edge_traces = []
    
    # Draw edges
    for edge in edges_list:
        x0, y0 = node_positions[edge["source"]]
        x1, y1 = node_positions[edge["target"]]
        
        # Color based on citation and GNN score
        if edge["cited"]:
            color = "red"
            width = 3
        elif edge["gnn_score"] is not None:
            # Color by GNN score
            score = edge["gnn_score"]
            color = f"rgba(0, 150, 255, {score})"
            width = 2
        else:
            color = "lightgray"
            width = 1
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"[Path {edge['path_idx']}] {edge['relation']}" + (f" (GNN: {edge['gnn_score']:.3f})" if edge['gnn_score'] else ""),
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Draw nodes
    node_x = [node_positions[node][0] for node in nodes]
    node_y = [node_positions[node][1] for node in nodes]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=nodes,
        textposition="top center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Retrieved Knowledge Graph Paths",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def fmt_path(p):
    """Format path for text display"""
    try:
        if not p:
            return ""
        
        # New dict format with triples
        if isinstance(p, dict):
            if "text" in p and p["text"]:
                return p["text"]
            if "triples" in p:
                parts = [f"{t['src']} —[{t['relation']}]→ {t['dst']}" for t in p["triples"]]
                return " | ".join(parts)
        
        # Legacy format: list of triples
        if isinstance(p[0], (list, tuple)):
            parts = []
            for e in p:
                if len(e) == 3:
                    h, r, t = e
                    parts.append(f"{h} —[{r}]→ {t}")
                elif len(e) == 2:
                    u, v = e
                    parts.append(f"{u} → {v}")
                else:
                    parts.append(str(e))
            return " | ".join(parts) if any("—[" in x for x in parts) else " → ".join(parts)
        
        # List of node IDs
        return " → ".join(map(str, p))
    except Exception:
        return str(p)

# --- Sidebar: API & settings ---
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    api_base = st.text_input("API Base URL", API_URL)

    # System health
    st.markdown("### 📊 System Health")
    cols = st.columns(3)
    
    with cols[0]:
        try:
            h = requests.get(f"{api_base}/health", timeout=3)
            if h.json().get("status") == "ok":
                st.success("API ✓")
            else:
                st.warning("API ⚠️")
        except:
            st.error("API ✗")
    
    with cols[1]:
        # Check Neo4j (if enabled)
        use_neo4j = os.environ.get("USE_NEO4J", "false").lower() == "true"
        if use_neo4j:
            st.info("Neo4j ✓")
        else:
            st.caption("Neo4j -")
    
    with cols[2]:
        # Check GNN (if enabled)
        use_gnn = os.environ.get("USE_GNN", "false").lower() == "true"
        if use_gnn:
            st.info("GNN ✓")
        else:
            st.caption("GNN -")

    st.markdown("### 🔍 Retrieval Settings")
    topk = st.number_input("Top-K paths", 1, 50, 5)
    max_hops = st.number_input("Max hops", 1, 10, 3)
    neighbor_expand = st.number_input("Neighborhood hops", 0, 5, 1)
    
    st.markdown("### 🤖 Model Settings")
    # Models (best-effort)
    try:
        with st.spinner("Fetching models..."):
            r = requests.get(f"{api_base}/models", timeout=5)
            models = r.json() if (r.status_code == 200 and isinstance(r.json(), list) and r.json()) else ["ollama/llama3", "ollama/llama3.2:3B"]
    except Exception:
        models = ["ollama/llama3", "ollama/llama3.2:3B"]
    model = st.selectbox("LLM Model", models, index=0)

    dry_run = st.checkbox("Dry run (skip LLM)", value=False)
    
    st.markdown("### 📈 Display Options")
    path_limit = st.selectbox("Paths to display", options=[10, 20, 50, "All"], index=0)
    show_graph_viz = st.checkbox("Show graph visualization", value=True)
    show_gnn_scores = st.checkbox("Show GNN confidence scores", value=True)

st.markdown("---")

# Main question input
st.markdown("## 💬 Ask a Question")
q = st.text_input("Question", value="", placeholder="e.g., Who co-authored a paper with Carol?")

# --- Ask button ---
if st.button("🚀 Ask", type="primary"):
    if not q.strip():
        st.warning("Please enter a question")
    else:
        try:
            payload = {
                "question": q,
                "topk_paths": int(topk),
                "max_hops": int(max_hops),
                "neighbor_expand": int(neighbor_expand),
                "model": model,
            }
            url = f"{api_base}/ask?dry_run={'true' if dry_run else 'false'}"
            
            with st.spinner("🔄 Processing your question..."):
                resp = requests.post(url, json=payload, timeout=60)
            
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("error"):
                st.error(f"❌ Error: {data.get('error')}")
            else:
                ctx = data.get("ctx", {}) or {}
                paths = ctx.get("paths", [])
                citations = data.get("citations", [])
                reasoning_trace = data.get("reasoning_trace")

                # ----- Answer Section -----
                if not dry_run:
                    st.markdown("## ✨ Answer")
                    answer = data.get("answer", "")
                    
                    # Highlight citations in answer
                    if citations:
                        for cite in citations:
                            pattern = f"\\[Path {cite['path_index']}\\]"
                            answer = re.sub(
                                pattern,
                                f"**[Path {cite['path_index']}]**",
                                answer,
                                flags=re.IGNORECASE
                            )
                    
                    st.markdown(answer)
                    
                    # Source info
                    source = data.get("source", "unknown")
                    source_icon = "🧮" if "gnn" in source.lower() else "🔍"
                    st.caption(f"{source_icon} Source: {source}")

                # ----- Citations -----
                if citations:
                    st.markdown("### 📌 Cited Sources")
                    for cite in citations:
                        with st.expander(f"Path {cite['path_index']}" + (f" (Confidence: {cite['confidence']:.3f})" if cite.get('confidence') else "")):
                            st.markdown(cite.get("text", ""))

                # ----- Reasoning Trace -----
                if reasoning_trace:
                    with st.expander("🔍 Reasoning Process", expanded=False):
                        visualize_reasoning_trace(reasoning_trace)

                # ----- Path Visualization -----
                if paths and show_graph_viz:
                    st.markdown("### 🕸️ Retrieved Knowledge Graph")
                    visualize_path_graph(paths, citations)

                # ----- Paths List -----
                with st.expander("📋 Retrieved Paths", expanded=True):
                    if not paths:
                        st.info("No paths returned")
                    else:
                        shown = paths if path_limit == "All" else paths[:int(path_limit)]
                        for i, p in enumerate(shown, 1):
                            is_cited = any(c["path_index"] == i for c in citations)
                            prefix = "⭐" if is_cited else "•"
                            
                            path_text = fmt_path(p)
                            
                            # Show GNN score if available
                            if show_gnn_scores and isinstance(p, dict):
                                gnn_score = p.get("gnn_score")
                                if gnn_score is not None:
                                    path_text += f" _(GNN: {gnn_score:.3f})_"
                            
                            st.markdown(f"{prefix} **Path {i}:** {path_text}")

                # ----- Context Details -----
                with st.expander("🔬 Context & Debug Info", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Detected Entities (Seeds)**")
                        st.write(ctx.get("seeds", []))
                        
                        st.markdown("**Node Notes**")
                        for note in ctx.get("node_notes", [])[:10]:
                            st.caption(note)
                    
                    with col2:
                        st.markdown("**Retrieval Metadata**")
                        st.json({
                            "retrieval_method": ctx.get("retrieval_method", "unknown"),
                            "paths_returned": len(paths),
                            "paths_cited": len(citations),
                            "source": data.get("source"),
                        })
                        
                        if st.checkbox("Show full context JSON"):
                            st.json(ctx)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)
