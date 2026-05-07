from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


class BaseDatasetBuilder(ABC):
    def __init__(self, config: dict[str, Any], root: str | Path = "outputs/datasets") -> None:
        self.config = config
        self.root = Path(config.get("root", root))
        self.seed = int(config.get("seed", 42))

    @abstractmethod
    def build(self) -> dict[str, list[nx.Graph]]:
        raise NotImplementedError

    def split_graphs(self, graphs: list[nx.Graph]) -> dict[str, list[nx.Graph]]:
        split = self.config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        total = len(graphs)
        n_train = int(total * float(split.get("train", 0.8)))
        n_val = int(total * float(split.get("val", 0.1)))
        return {
            "train": graphs[:n_train],
            "val": graphs[n_train : n_train + n_val],
            "test": graphs[n_train + n_val :],
        }

    def normalize_graph(self, graph: nx.Graph) -> nx.Graph:
        g = nx.Graph(graph)  # simple undirected copy; removes multi-edge semantics
        if self.config.get("remove_self_loops", True):
            g.remove_edges_from(nx.selfloop_edges(g))
        g = nx.convert_node_labels_to_integers(g)
        return g


def graph_statistics(graphs: list[nx.Graph]) -> dict[str, float | int]:
    if not graphs:
        return {"num_graphs": 0}
    n = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    m = np.asarray([g.number_of_edges() for g in graphs], dtype=float)
    density = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs], dtype=float)
    clustering = []
    connected = []
    planar = []
    for g in graphs:
        try:
            clustering.append(float(nx.average_clustering(g)))
        except Exception:
            clustering.append(0.0)
        try:
            connected.append(float(nx.is_connected(g)) if g.number_of_nodes() > 0 else 0.0)
        except Exception:
            connected.append(0.0)
        try:
            planar.append(float(nx.check_planarity(g)[0]))
        except Exception:
            planar.append(0.0)
    return {
        "num_graphs": int(len(graphs)),
        "num_nodes_mean": float(n.mean()),
        "num_nodes_std": float(n.std(ddof=0)),
        "num_edges_mean": float(m.mean()),
        "num_edges_std": float(m.std(ddof=0)),
        "density_mean": float(density.mean()),
        "density_std": float(density.std(ddof=0)),
        "avg_clustering_mean": float(np.mean(clustering)),
        "connected_rate": float(np.mean(connected)),
        "planarity_rate": float(np.mean(planar)),
    }
