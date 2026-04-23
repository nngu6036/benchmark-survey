from __future__ import annotations
import networkx as nx
import numpy as np

class RandomGINPlaceholder:
    def __init__(self, feature_dim: int = 128, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)

    def encode(self, graph: nx.Graph) -> np.ndarray:
        base = np.array([
            graph.number_of_nodes(),
            graph.number_of_edges(),
            nx.density(graph),
            sum(nx.triangles(graph).values()) / 3.0,
        ], dtype=float)
        proj = self.rng.normal(size=(base.shape[0], self.feature_dim))
        return base @ proj
