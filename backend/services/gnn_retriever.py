# backend/services/gnn_retriever.py
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import numpy as np


class GNNEncoder(nn.Module):
    """
    Graph Convolutional Network for encoding graph structure.
    Uses 2-layer GCN to capture node relationships and neighborhood structure.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GNNRetriever:
    """
    GNN-based path retriever that scores paths using graph structure.
    Complements keyword/BM25 retrieval with structural graph knowledge.
    
    Key Features:
    - Encodes nodes using graph structure (not just text)
    - Scores paths based on node embeddings and edge importance
    - Can be pre-trained on graph structure or used zero-shot
    """

    def __init__(
        self,
        graph_data: Optional[Data] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = None
        self.data = graph_data
        self.node_id_to_idx = {}
        self.embeddings = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def build_from_networkx(
        self, G: nx.Graph, node_features: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Build PyG Data object from NetworkX graph.

        Args:
            G: NetworkX graph
            node_features: Optional node feature dict. If None, uses degree as feature.
        """
        # Map node IDs to indices
        self.node_id_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        num_nodes = len(self.node_id_to_idx)

        # Create edge index for PyG
        edge_list = []
        for u, v in G.edges():
            edge_list.append([self.node_id_to_idx[u], self.node_id_to_idx[v]])
            edge_list.append([self.node_id_to_idx[v], self.node_id_to_idx[u]])  # Undirected

        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create node features
        if node_features is None:
            # Use degree centrality as simple feature
            degrees = dict(G.degree())
            x = torch.tensor(
                [[degrees.get(node, 0)] for node in G.nodes()],
                dtype=torch.float,
            )
        else:
            # Stack provided features
            x = torch.stack([
                torch.tensor(node_features.get(node, [0.0]), dtype=torch.float)
                for node in G.nodes()
            ])

        self.data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        self.data = self.data.to(self.device)

        # Initialize model
        in_channels = x.shape[1]
        self.model = GNNEncoder(in_channels=in_channels).to(self.device)

    def train_embeddings(self, epochs: int = 50, lr: float = 0.01):
        """
        Pre-train GNN embeddings using unsupervised learning.
        Uses link prediction as pretext task.
        """
        if self.model is None or self.data is None:
            raise ValueError("Model and data must be initialized first")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"🔄 Training GNN embeddings for {epochs} epochs...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            embeddings = self.model(self.data.x, self.data.edge_index)

            # Link prediction loss (predict edges exist)
            edge_index = self.data.edge_index
            pos_score = (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

            # Negative sampling
            neg_edge_index = self._negative_sampling(edge_index, self.data.num_nodes)
            neg_score = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Cache embeddings
        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model(self.data.x, self.data.edge_index)

        print("✅ GNN training complete")

    def _negative_sampling(self, edge_index, num_nodes, num_neg_samples=None):
        """Sample negative edges for link prediction"""
        if num_neg_samples is None:
            num_neg_samples = edge_index.shape[1]

        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst:
                neg_edges.append([src, dst])

        return torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)

    def get_embeddings(self) -> torch.Tensor:
        """Get or compute node embeddings"""
        if self.embeddings is not None:
            return self.embeddings

        if self.model is None or self.data is None:
            raise ValueError("Model not initialized")

        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model(self.data.x, self.data.edge_index)

        return self.embeddings

    def score_paths(
        self, paths: List[Dict[str, Any]], question_embedding: Optional[torch.Tensor] = None
    ) -> List[float]:
        """
        Score paths using GNN embeddings and graph structure.

        Args:
            paths: List of path dictionaries with 'nodes' or node IDs
            question_embedding: Optional query embedding for relevance scoring

        Returns:
            List of scores (higher = better)
        """
        embeddings = self.get_embeddings()
        scores = []

        for path in paths:
            # Extract node IDs from path
            node_ids = self._extract_node_ids(path)

            # Get embeddings for nodes in path
            node_indices = [
                self.node_id_to_idx[nid]
                for nid in node_ids
                if nid in self.node_id_to_idx
            ]

            if not node_indices:
                scores.append(0.0)
                continue

            path_embeddings = embeddings[node_indices]

            # Score based on:
            # 1. Path coherence (how well connected nodes are)
            # 2. Node centrality (important nodes score higher)
            # 3. Query relevance (if question embedding provided)

            # Coherence: average pairwise similarity of adjacent nodes
            coherence = 0.0
            if len(path_embeddings) > 1:
                for i in range(len(path_embeddings) - 1):
                    sim = F.cosine_similarity(
                        path_embeddings[i].unsqueeze(0),
                        path_embeddings[i + 1].unsqueeze(0),
                    ).item()
                    coherence += sim
                coherence /= (len(path_embeddings) - 1)

            # Centrality: average embedding norm (captures node importance)
            centrality = path_embeddings.norm(dim=1).mean().item()

            # Combine scores
            score = 0.6 * coherence + 0.4 * centrality

            # Optional: query relevance
            if question_embedding is not None:
                path_avg = path_embeddings.mean(dim=0)
                relevance = F.cosine_similarity(
                    path_avg.unsqueeze(0), question_embedding.unsqueeze(0)
                ).item()
                score = 0.4 * score + 0.6 * relevance

            scores.append(score)

        return scores

    def _extract_node_ids(self, path: Dict[str, Any]) -> List[str]:
        """Extract node IDs from various path formats"""
        if "nodes" in path:
            # Format: {'nodes': [{'id': 'x'}, ...]}
            return [n["id"] if isinstance(n, dict) else n for n in path["nodes"]]
        elif "triples" in path:
            # Format: {'triples': [{'src': 'x', 'dst': 'y'}, ...]}
            nodes = []
            for triple in path["triples"]:
                if isinstance(triple, dict):
                    nodes.extend([triple.get("src"), triple.get("dst")])
                elif isinstance(triple, (list, tuple)) and len(triple) >= 2:
                    nodes.extend([triple[0], triple[2]])
            return [n for n in nodes if n]
        elif "path" in path:
            # Format: {'path': ['x', 'y', 'z']}
            return path["path"]
        return []

    def save_model(self, path: str):
        """Save GNN model and mappings"""
        save_dict = {
            "model_state": self.model.state_dict() if self.model else None,
            "node_id_to_idx": self.node_id_to_idx,
            "embeddings": self.embeddings.cpu() if self.embeddings is not None else None,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        print(f"✅ GNN model saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained GNN model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.node_id_to_idx = checkpoint["node_id_to_idx"]

        if checkpoint["embeddings"] is not None:
            self.embeddings = checkpoint["embeddings"].to(self.device)

        # Reconstruct model if state dict available
        if checkpoint["model_state"]:
            in_channels = self.embeddings.shape[1] if self.embeddings is not None else 1
            self.model = GNNEncoder(in_channels=in_channels).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

        print(f"✅ GNN model loaded from {path}")


# Global instance (lazy initialization)
_gnn_retriever: Optional[GNNRetriever] = None


def get_gnn_retriever() -> Optional[GNNRetriever]:
    """Get or initialize global GNN retriever"""
    global _gnn_retriever

    if not os.environ.get("USE_GNN", "false").lower() == "true":
        return None

    if _gnn_retriever is None:
        model_path = os.environ.get("GNN_MODEL_PATH", "models/gnn_retriever.pt")
        if os.path.exists(model_path):
            _gnn_retriever = GNNRetriever(model_path=model_path)
        else:
            print("⚠️ GNN model not found. Run scripts/train_gnn.py to train.")

    return _gnn_retriever
