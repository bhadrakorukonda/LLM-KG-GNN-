#!/usr/bin/env python3
"""
Load knowledge graph data into Neo4j from TSV files.
Usage: python scripts/load_neo4j.py
"""
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.neo4j_store import Neo4jStore


def main():
    # Get file paths
    repo_root = Path(__file__).resolve().parents[1]
    edges_file = os.environ.get("KG_EDGES", str(repo_root / "data" / "kg_edges.tsv"))
    node_texts_file = os.environ.get(
        "KG_NODE_TEXTS", str(repo_root / "data" / "node_texts.jsonl")
    )

    print("🔄 Loading data into Neo4j...")
    print(f"   Edges: {edges_file}")
    print(f"   Nodes: {node_texts_file}")

    # Initialize store
    store = Neo4jStore()

    # Health check
    if not store.health_check():
        print("❌ Cannot connect to Neo4j. Make sure it's running:")
        print("   docker-compose up neo4j")
        sys.exit(1)

    # Load data
    store.load_from_tsv(edges_file, node_texts_file)

    # Verify
    store.connect()
    with store.driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) as cnt").single()["cnt"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()[
            "cnt"
        ]
        print(f"✅ Loaded: {node_count} nodes, {edge_count} edges")

    store.close()


if __name__ == "__main__":
    main()
