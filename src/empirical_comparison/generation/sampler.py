from __future__ import annotations

import time
from typing import Any, Callable

from empirical_comparison.registry import get_model_class
from empirical_comparison.utils.progress import progress_bar, update_progress
from empirical_comparison.utils.numerics import assert_finite_graphs


def sample_graphs(
    model_name: str,
    model_cfg: dict[str, Any],
    num_graphs: int,
    seed: int = 0,
    *,
    show_progress: bool = False,
    progress_desc: str | None = None,
):
    cls = get_model_class(model_name)
    model = cls(model_cfg)
    start = time.perf_counter()
    model.load()
    desc = progress_desc or f"Sampling {model_name}"
    with progress_bar(total=int(num_graphs), desc=desc, unit="graph", enabled=show_progress) as raw_update:
        completed = 0

        def progress_callback(amount: int) -> None:
            nonlocal completed
            amount = max(0, int(amount))
            remaining = max(0, int(num_graphs) - completed)
            inc = min(amount, remaining)
            if inc > 0:
                completed += inc
                raw_update(inc)

        try:
            graphs = model.sample(num_graphs=num_graphs, seed=seed, progress_callback=progress_callback)
        except TypeError as exc:
            # Backward compatibility for external wrapper implementations that
            # still expose sample(num_graphs, seed) only.  The local wrappers all
            # accept progress_callback.
            if "progress_callback" not in str(exc):
                raise
            graphs = model.sample(num_graphs=num_graphs, seed=seed)
        update_progress(progress_callback, min(len(graphs), int(num_graphs)) - completed)
    assert_finite_graphs(graphs, context=f"{model_name}.sample output")
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
