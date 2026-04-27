from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


def _safe_triangle_count(graph: nx.Graph) -> float:
    try:
        return float(sum(nx.triangles(graph).values()) / 3.0)
    except Exception:
        return 0.0


def _safe_avg_clustering(graph: nx.Graph) -> float:
    try:
        return float(nx.average_clustering(graph))
    except Exception:
        return 0.0


@dataclass
class RandomGINPlaceholder:
    """
    A lightweight placeholder for a learned-feature evaluator.

    Important:
    - This is NOT a trained GIN.
    - It is a fixed random feature map over simple graph statistics.
    - The projection matrix is sampled once at initialization and reused
      for every graph, so all graphs are embedded in the same feature space.
    """

    feature_dim: int = 128
    seed: int = 42
    normalize_output: bool = True

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

        # Base feature vector length:
        # 0: log(1 + num_nodes)
        # 1: log(1 + num_edges)
        # 2: density
        # 3: log(1 + triangle_count)
        # 4: average clustering
        # 5: average degree
        self.base_dim = 6

        # Fixed random projection shared by all graphs
        self.proj = self.rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.base_dim),
            size=(self.base_dim, self.feature_dim),
        )

        # Fixed random bias
        self.bias = self.rng.normal(
            loc=0.0,
            scale=0.1,
            size=(self.feature_dim,),
        )

    def _base_features(self, graph: nx.Graph) -> np.ndarray:
        n = float(graph.number_of_nodes())
        e = float(graph.number_of_edges())

        density = float(nx.density(graph)) if n > 1 else 0.0
        tri = _safe_triangle_count(graph)
        avg_clust = _safe_avg_clustering(graph)
        avg_deg = (2.0 * e / n) if n > 0 else 0.0

        # Use log scaling for count-like quantities to reduce magnitude blow-up
        feat = np.array(
            [
                np.log1p(n),
                np.log1p(e),
                density,
                np.log1p(tri),
                avg_clust,
                avg_deg,
            ],
            dtype=np.float64,
        )
        return feat

    def encode(self, graph: nx.Graph) -> np.ndarray:
        base = self._base_features(graph)

        # Fixed projection into shared feature space
        z = base @ self.proj + self.bias

        # Simple nonlinearity to mimic a shallow neural embedding map
        z = np.tanh(z)

        if self.normalize_output:
            norm = np.linalg.norm(z)
            if norm > 1e-12:
                z = z / norm

        return z.astype(np.float64)