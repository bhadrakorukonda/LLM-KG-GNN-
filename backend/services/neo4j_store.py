# backend/services/neo4j_store.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase


class Neo4jStore:
    """
    Neo4j graph store for GRAIL-LM.
    Handles connection to Neo4j and subgraph retrieval for Graph-RAG.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j"""
        if not self.driver:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_subgraph(
        self, seed_ids: List[str], max_hops: int = 3, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k-hop subgraph starting from seed nodes.
        Returns list of paths with proper source tracing.

        Args:
            seed_ids: Starting node IDs
            max_hops: Maximum path length
            limit: Maximum number of paths to return

        Returns:
            List of path dictionaries with nodes, relationships, and metadata
        """
        self.connect()

        query = """
        MATCH path = (start)-[*1..{max_hops}]-(end)
        WHERE start.id IN $seeds
        WITH path, length(path) as pathLen
        ORDER BY pathLen
        LIMIT $limit
        RETURN path,
               [node IN nodes(path) | {id: node.id, text: node.text}] as node_list,
               [rel IN relationships(path) | {type: type(rel), source: startNode(rel).id, target: endNode(rel).id}] as edge_list,
               pathLen
        """.format(max_hops=max_hops)

        with self.driver.session() as session:
            result = session.run(query, seeds=seed_ids, limit=limit)
            paths = []
            for record in result:
                paths.append({
                    "nodes": record["node_list"],
                    "edges": record["edge_list"],
                    "length": record["pathLen"],
                    "path": record["path"],
                })
            return paths

    def get_neighborhood(
        self, node_id: str, relation_types: Optional[List[str]] = None, hops: int = 1
    ) -> Dict[str, Any]:
        """
        Get immediate neighborhood of a node with optional relation filtering.

        Args:
            node_id: Center node ID
            relation_types: Optional list of relation types to filter
            hops: Number of hops (default 1)

        Returns:
            Dictionary with neighbors and their relationships
        """
        self.connect()

        rel_filter = (
            f"AND type(r) IN {relation_types}" if relation_types else ""
        )

        query = f"""
        MATCH (n {{id: $node_id}})-[r*1..{hops}]-(neighbor)
        {rel_filter}
        RETURN DISTINCT neighbor.id as neighbor_id,
               neighbor.text as text,
               type(r[0]) as relation,
               length(r) as distance
        ORDER BY distance
        """

        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            neighbors = []
            for record in result:
                neighbors.append({
                    "id": record["neighbor_id"],
                    "text": record["text"],
                    "relation": record["relation"],
                    "distance": record["distance"],
                })
            return {"node_id": node_id, "neighbors": neighbors}

    def shortest_paths_multi_seed(
        self,
        seed_ids: List[str],
        max_hops: int = 3,
        topk: int = 10,
        allowed_relations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find shortest paths between all pairs of seed nodes.
        This enables multi-hop reasoning across entities.

        Args:
            seed_ids: List of seed node IDs
            max_hops: Maximum path length
            topk: Top K paths to return
            allowed_relations: Optional relation type filter

        Returns:
            List of shortest paths with full traceability
        """
        self.connect()

        rel_filter = (
            f"AND ALL(r IN relationships(path) WHERE type(r) IN {allowed_relations})"
            if allowed_relations
            else ""
        )

        query = f"""
        MATCH path = allShortestPaths((start)-[*..{max_hops}]-(end))
        WHERE start.id IN $seeds
          AND end.id IN $seeds
          AND start <> end
          {rel_filter}
        WITH path, 
             [node IN nodes(path) | {{id: node.id, text: node.text}}] as nodes,
             [rel IN relationships(path) | {{type: type(rel), source: startNode(rel).id, target: endNode(rel).id}}] as edges,
             length(path) as pathLen
        ORDER BY pathLen
        LIMIT $topk
        RETURN nodes, edges, pathLen
        """

        with self.driver.session() as session:
            result = session.run(query, seeds=seed_ids, topk=topk)
            paths = []
            for record in result:
                # Format for compatibility with existing pipeline
                path_dict = {
                    "nodes": record["nodes"],
                    "edges": record["edges"],
                    "length": record["pathLen"],
                    "triples": [
                        {
                            "src": edge["source"],
                            "relation": edge["type"],
                            "dst": edge["target"],
                        }
                        for edge in record["edges"]
                    ],
                    "text": self._format_path_text(record["nodes"], record["edges"]),
                }
                paths.append(path_dict)
            return paths

    def _format_path_text(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Format path as readable text for LLM context"""
        parts = []
        for i, edge in enumerate(edges):
            src_text = next(n["text"] for n in nodes if n["id"] == edge["source"])
            dst_text = next(n["text"] for n in nodes if n["id"] == edge["target"])
            parts.append(f"{src_text} —[{edge['type']}]→ {dst_text}")
        return " | ".join(parts) if parts else ""

    def load_from_tsv(self, edges_file: str, node_texts_file: str):
        """
        Load graph data from TSV files into Neo4j.
        Creates nodes and relationships from kg_edges.tsv and node_texts.jsonl

        Args:
            edges_file: Path to kg_edges.tsv (head, relation, tail)
            node_texts_file: Path to node_texts.jsonl (id, text)
        """
        import json

        self.connect()

        with self.driver.session() as session:
            # Clear existing data (use with caution!)
            session.run("MATCH (n) DETACH DELETE n")

            # Load node texts
            node_texts = {}
            if os.path.exists(node_texts_file):
                with open(node_texts_file, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            nid = data.get("id")
                            text = data.get("text") or data.get("desc") or ""
                            if nid:
                                node_texts[nid] = text
                        except:
                            pass

            # Create nodes
            for node_id, text in node_texts.items():
                session.run(
                    "MERGE (n:Entity {id: $id}) SET n.text = $text",
                    id=node_id,
                    text=text,
                )

            # Load edges
            if os.path.exists(edges_file):
                with open(edges_file, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("head"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            head, relation, tail = parts[0], parts[1], parts[2]

                            # Ensure nodes exist
                            session.run(
                                "MERGE (h:Entity {id: $head}) MERGE (t:Entity {id: $tail})",
                                head=head,
                                tail=tail,
                            )

                            # Create relationship (Neo4j requires valid relationship type names)
                            rel_type = relation.upper().replace("-", "_").replace(" ", "_")
                            session.run(
                                f"MATCH (h:Entity {{id: $head}}), (t:Entity {{id: $tail}}) "
                                f"MERGE (h)-[r:{rel_type}]->(t)",
                                head=head,
                                tail=tail,
                            )

            print(f"✅ Loaded {len(node_texts)} nodes and edges into Neo4j")

    def health_check(self) -> bool:
        """Check if Neo4j is reachable"""
        try:
            self.connect()
            with self.driver.session() as session:
                result = session.run("RETURN 1 as health")
                return result.single()["health"] == 1
        except Exception as e:
            print(f"Neo4j health check failed: {e}")
            return False


# Global instance
neo4j_store = Neo4jStore()
