from __future__ import annotations
import networkx as nx
from empirical_comparison.models.base import BaseGenerator

class GruMWrapper(BaseGenerator):
    @property
    def name(self) -> str:
        return "grum"

    def load(self) -> None:
        return None

    def train(self, train_graphs, val_graphs=None) -> None:
        return None

    def sample(self, num_graphs: int, seed: int = 0):
        return [nx.erdos_renyi_graph(32, 0.1, seed=seed + i) for i in range(num_graphs)]
