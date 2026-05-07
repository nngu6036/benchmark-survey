from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

from empirical_comparison.graphs.attributes import apply_empirical_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.io import load_pickle, save_pickle


class DummyGraphGenerator(BaseGenerator):
    """Lightweight baseline and smoke-test wrapper.

    It estimates simple graph-size and density statistics from training graphs
    and samples Erdos--Renyi graphs with similar average size/density.  This is
    not intended as a competitive model; it verifies the benchmark pipeline.
    """

    supports_training = True
    supports_sampling = True
    supports_node_features = True
    supports_edge_features = True
    supports_node_labels = True
    supports_edge_labels = True
    supports_graph_labels = True
    supports_constraints = False
    supports_variable_size = True
    supports_featureless_graphs = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.num_nodes = int(config.get("num_nodes", 64))
        self.edge_prob = float(config.get("edge_prob", 0.05))
        self.variable_size = bool(config.get("variable_size", False))
        self.node_std = float(config.get("node_std", 0.0))
        self.checkpoint_path = config.get("checkpoint_path")
        self.attr_schema = normalize_schema(config)
        self.attr_stats = None
        if isinstance(config.get("graph_attribute_stats"), dict):
            self.attr_stats = config.get("graph_attribute_stats")

    @property
    def name(self) -> str:
        return "dummy"

    def load(self) -> None:
        if not self.checkpoint_path:
            return None
        path = Path(self.checkpoint_path)
        if not path.exists():
            return None
        state = load_pickle(path)
        if isinstance(state, dict):
            self.num_nodes = int(state.get("num_nodes", self.num_nodes))
            self.edge_prob = float(state.get("edge_prob", self.edge_prob))
            self.variable_size = bool(state.get("variable_size", self.variable_size))
            self.node_std = float(state.get("node_std", self.node_std))
            self.attr_stats = state.get("graph_attribute_stats", self.attr_stats)
        return None

    def train(self, train_graphs: Sequence[nx.Graph], val_graphs=None, test_graphs=None) -> None:
        if train_graphs:
            nodes = np.asarray([g.number_of_nodes() for g in train_graphs], dtype=float)
            densities = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in train_graphs], dtype=float)
            self.num_nodes = max(1, int(round(float(nodes.mean()))))
            self.node_std = float(nodes.std(ddof=0))
            self.edge_prob = float(np.clip(densities.mean(), 0.0, 1.0))
            self.attr_stats = fit_attribute_statistics(list(train_graphs), self.attr_schema).to_dict()
        if self.checkpoint_path:
            save_pickle(
                {
                    "num_nodes": self.num_nodes,
                    "edge_prob": self.edge_prob,
                    "variable_size": self.variable_size,
                    "node_std": self.node_std,
                    "config": self.config,
                    "graph_attribute_stats": self.attr_stats,
                },
                self.checkpoint_path,
                force=True,
            )

    def sample(self, num_graphs: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs: list[nx.Graph] = []
        for _ in range(num_graphs):
            if self.variable_size and self.node_std > 0:
                n = max(1, int(round(rng.normal(self.num_nodes, self.node_std))))
            else:
                n = self.num_nodes
            g = nx.gnp_random_graph(n=n, p=self.edge_prob, seed=int(rng.integers(0, 2**31 - 1)))
            graphs.append(nx.convert_node_labels_to_integers(g))
        if self.attr_stats:
            graphs = apply_empirical_attributes(graphs, self.attr_stats, seed=seed, overwrite=True)
        return graphs
