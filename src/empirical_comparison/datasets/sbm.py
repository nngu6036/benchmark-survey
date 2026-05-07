from __future__ import annotations

import networkx as nx
import numpy as np

from empirical_comparison.datasets.base import BaseDatasetBuilder


class SBMDatasetBuilder(BaseDatasetBuilder):
    def build(self):
        n = int(self.config.get("num_nodes", 64))
        b = int(self.config.get("num_blocks", 4))
        if b <= 0:
            raise ValueError("num_blocks must be positive")
        base = n // b
        sizes = [base] * b
        sizes[-1] += n - sum(sizes)
        p_in = float(self.config.get("p_in", 0.25))
        p_out = float(self.config.get("p_out", 0.02))
        total = int(self.config.get("num_graphs", 128))
        probs = [[p_in if i == j else p_out for j in range(b)] for i in range(b)]
        rng = np.random.default_rng(self.seed)
        graphs = [
            self.normalize_graph(
                nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(0, 2**31 - 1)))
            )
            for _ in range(total)
        ]
        return self.split_graphs(graphs)
