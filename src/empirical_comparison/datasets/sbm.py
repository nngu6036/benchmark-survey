from __future__ import annotations
import networkx as nx
from empirical_comparison.datasets.base import BaseDatasetBuilder

class SBMDatasetBuilder(BaseDatasetBuilder):
    def build(self):
        n = self.config.get("num_nodes", 64)
        b = self.config.get("num_blocks", 4)
        per_block = n // b
        sizes = [per_block] * b
        p_in = self.config.get("p_in", 0.25)
        p_out = self.config.get("p_out", 0.02)
        total = self.config.get("num_graphs", 128)
        probs = [[p_in if i == j else p_out for j in range(b)] for i in range(b)]
        graphs = [nx.stochastic_block_model(sizes, probs, seed=i) for i in range(total)]
        split = self.config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        n_train = int(total * split["train"])
        n_val = int(total * split["val"])
        return {"train": graphs[:n_train], "val": graphs[n_train:n_train+n_val], "test": graphs[n_train+n_val:]}
