from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from empirical_comparison.datasets.base import BaseDatasetBuilder


class PlanarDatasetBuilder(BaseDatasetBuilder):
    def build(self):
        total = int(self.config.get("num_graphs", 128))
        n = int(self.config.get("num_nodes", 64))
        rng = np.random.default_rng(self.seed)
        graphs = [
            self.normalize_graph(self._sample_planar_graph(n, seed=int(rng.integers(0, 2**31 - 1))))
            for _ in range(total)
        ]
        return self.split_graphs(graphs)

    @staticmethod
    def _sample_planar_graph(n: int, seed: int) -> nx.Graph:
        if n <= 0:
            raise ValueError("num_nodes must be positive")
        rng = np.random.default_rng(seed)
        pts = rng.random((n, 2))
        g = nx.Graph()
        g.add_nodes_from(range(n))
        if n < 3:
            return g
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            for i in range(3):
                u = int(simplex[i])
                v = int(simplex[(i + 1) % 3])
                g.add_edge(u, v)
        return g
