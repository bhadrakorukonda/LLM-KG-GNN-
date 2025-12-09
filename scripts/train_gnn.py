#!/usr/bin/env python3
"""
Train GNN embeddings for graph-based retrieval.
Usage: python scripts/train_gnn.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.graph import graph_store
from backend.services.gnn_retriever import GNNRetriever


def main():
    print("🚀 Training GNN for Graph-RAG retrieval...")

    # Load graph
    print("📊 Loading graph from data files...")
    graph_store.ensure_loaded()
    G = graph_store.G

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # Initialize GNN retriever
    gnn = GNNRetriever(device="cpu")
    gnn.build_from_networkx(G)

    # Train embeddings
    epochs = int(os.environ.get("GNN_EPOCHS", "100"))
    gnn.train_embeddings(epochs=epochs, lr=0.01)

    # Save model
    model_path = Path(__file__).resolve().parents[1] / "models" / "gnn_retriever.pt"
    gnn.save_model(str(model_path))

    print("✅ GNN training complete!")
    print(f"   Model saved to: {model_path}")
    print("\nTo use GNN in retrieval, set environment variable:")
    print("   USE_GNN=true")


if __name__ == "__main__":
    main()
