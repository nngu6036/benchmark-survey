from __future__ import annotations

from pathlib import Path
from typing import Sequence
import time

import networkx as nx
import numpy as np

from empirical_comparison.graphs.attributes import apply_empirical_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.models.base import BaseGenerator
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.progress import update_progress
from empirical_comparison.utils.io import load_pickle, save_pickle


LOGGER = get_logger(__name__)


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
        self.detailed_logging = bool(config.get("detailed_logging", True))
        self.log_sample_every_n_graphs = max(1, int(config.get("log_sample_every_n_graphs", 1)))
        self._log(
            "initialized num_nodes=%d edge_prob=%.6f variable_size=%s checkpoint_path=%s",
            self.num_nodes,
            self.edge_prob,
            self.variable_size,
            self.checkpoint_path,
        )

    @property
    def name(self) -> str:
        return "dummy"

    def load(self) -> None:
        started_at = time.perf_counter()
        self._log("load_start checkpoint_path=%s", self.checkpoint_path)
        if not self.checkpoint_path:
            self._log("load_skip reason=no_checkpoint_path")
            return None
        path = Path(self.checkpoint_path)
        if not path.exists():
            self._log("load_skip reason=checkpoint_missing path=%s", path)
            return None
        state = load_pickle(path)
        if isinstance(state, dict):
            self.num_nodes = int(state.get("num_nodes", self.num_nodes))
            self.edge_prob = float(state.get("edge_prob", self.edge_prob))
            self.variable_size = bool(state.get("variable_size", self.variable_size))
            self.node_std = float(state.get("node_std", self.node_std))
            self.attr_stats = state.get("graph_attribute_stats", self.attr_stats)
        self._log(
            "load_end num_nodes=%d edge_prob=%.6f variable_size=%s node_std=%.6f duration=%.3fs",
            self.num_nodes,
            self.edge_prob,
            self.variable_size,
            self.node_std,
            time.perf_counter() - started_at,
        )
        return None

    def train(self, train_graphs: Sequence[nx.Graph], val_graphs=None, test_graphs=None) -> None:
        started_at = time.perf_counter()
        self._log(
            "train_start train_count=%d val_count=%s test_count=%s",
            len(train_graphs or []),
            None if val_graphs is None else len(val_graphs),
            None if test_graphs is None else len(test_graphs),
        )
        if train_graphs:
            nodes = np.asarray([g.number_of_nodes() for g in train_graphs], dtype=float)
            densities = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in train_graphs], dtype=float)
            self.num_nodes = max(1, int(round(float(nodes.mean()))))
            self.node_std = float(nodes.std(ddof=0))
            self.edge_prob = float(np.clip(densities.mean(), 0.0, 1.0))
            self.attr_stats = fit_attribute_statistics(list(train_graphs), self.attr_schema).to_dict()
            self._log(
                "fit_statistics nodes_mean=%.3f nodes_std=%.3f density_mean=%.6f density_min=%.6f density_max=%.6f",
                float(nodes.mean()),
                self.node_std,
                float(densities.mean()),
                float(densities.min()),
                float(densities.max()),
            )
        if self.checkpoint_path:
            self._log("saving_checkpoint path=%s", self.checkpoint_path)
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
        self._log("train_end duration=%.3fs", time.perf_counter() - started_at)

    def sample(self, num_graphs: int, seed: int = 0, progress_callback=None):
        started_at = time.perf_counter()
        self._log("sample_start num_graphs=%d seed=%d", num_graphs, seed)
        rng = np.random.default_rng(seed)
        graphs: list[nx.Graph] = []
        for idx in range(num_graphs):
            if self.variable_size and self.node_std > 0:
                n = max(1, int(round(rng.normal(self.num_nodes, self.node_std))))
            else:
                n = self.num_nodes
            g = nx.gnp_random_graph(n=n, p=self.edge_prob, seed=int(rng.integers(0, 2**31 - 1)))
            graphs.append(nx.convert_node_labels_to_integers(g))
            if idx % self.log_sample_every_n_graphs == 0:
                self._log("sample_graph index=%d nodes=%d edges=%d", idx, g.number_of_nodes(), g.number_of_edges())
            update_progress(progress_callback, 1)
        if self.attr_stats:
            self._log("applying_empirical_attributes graph_count=%d", len(graphs))
            graphs = apply_empirical_attributes(graphs, self.attr_stats, seed=seed, overwrite=True)
        self._log("sample_end returned=%d duration=%.3fs", len(graphs), time.perf_counter() - started_at)
        return graphs

    def _log(self, message: str, *args) -> None:
        if self.detailed_logging:
            LOGGER.info("DummyGraphGenerator " + message, *args)
