from __future__ import annotations
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from empirical_comparison.datasets.base import BaseDatasetBuilder

class PlanarDatasetBuilder(BaseDatasetBuilder):
    def build(self):
        total = self.config.get("num_graphs", 128)
        n = self.config.get("num_nodes", 64)
        graphs = [self._sample_planar_graph(n, seed=i) for i in range(total)]
        split = self.config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        n_train = int(total * split["train"])
        n_val = int(total * split["val"])
        return {"train": graphs[:n_train], "val": graphs[n_train:n_train+n_val], "test": graphs[n_train+n_val:]}

    @staticmethod
    def _sample_planar_graph(n: int, seed: int) -> nx.Graph:
        rng = np.random.default_rng(seed)
        pts = rng.random((n, 2))
        tri = Delaunay(pts)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for simplex in tri.simplices:
            for i in range(3):
                u = int(simplex[i]); v = int(simplex[(i + 1) % 3])
                g.add_edge(u, v)
        return g
