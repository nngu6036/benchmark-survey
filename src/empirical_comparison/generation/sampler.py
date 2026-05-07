from __future__ import annotations

import time
from typing import Any

from empirical_comparison.registry import get_model_class


def sample_graphs(model_name: str, model_cfg: dict[str, Any], num_graphs: int, seed: int = 0):
    cls = get_model_class(model_name)
    model = cls(model_cfg)
    start = time.perf_counter()
    model.load()
    graphs = model.sample(num_graphs=num_graphs, seed=seed)
    elapsed = time.perf_counter() - start
    return graphs


def model_capabilities(model_name: str) -> dict[str, bool]:
    cls = get_model_class(model_name)
    if hasattr(cls, "capabilities"):
        return cls.capabilities()
    return {
        "supports_training": True,
        "supports_sampling": True,
        "supports_node_features": False,
        "supports_edge_features": False,
        "supports_node_labels": False,
        "supports_edge_labels": False,
        "supports_graph_labels": False,
        "supports_constraints": False,
        "supports_variable_size": True,
        "supports_featureless_graphs": True,
    }
