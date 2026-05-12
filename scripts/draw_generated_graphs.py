from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import sample_path
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_pickle
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _select_graphs(graphs: list[nx.Graph], *, num_graphs: int, seed: int | None, start_index: int) -> list[tuple[int, nx.Graph]]:
    if not graphs:
        raise ValueError("Generated sample file contains no graphs.")
    if num_graphs <= 0:
        raise ValueError("--num-graphs must be positive.")
    if seed is None:
        start = max(0, int(start_index))
        selected = list(range(start, min(start + num_graphs, len(graphs))))
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(graphs), size=min(num_graphs, len(graphs)), replace=False).tolist()
    return [(int(i), graphs[int(i)]) for i in selected]


def _layout(graph: nx.Graph, *, layout: str, seed: int) -> dict[Any, np.ndarray]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        node = next(iter(graph.nodes()))
        return {node: np.asarray([0.0, 0.0])}
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "spectral":
        try:
            return nx.spectral_layout(graph)
        except Exception:
            return nx.spring_layout(graph, seed=seed)
    if layout == "circular":
        return nx.circular_layout(graph)
    return nx.spring_layout(graph, seed=seed)


def _node_colors(graph: nx.Graph, label_attr: str) -> list[int]:
    labels = [graph.nodes[n].get(label_attr, graph.degree(n)) for n in graph.nodes()]
    keys = {str(v): i for i, v in enumerate(sorted({str(v) for v in labels}))}
    return [keys[str(v)] for v in labels]


def _draw_one(
    ax,
    graph: nx.Graph,
    *,
    title: str,
    layout: str,
    seed: int,
    node_label_attr: str,
    edge_label_attr: str,
    show_node_labels: bool,
    show_edge_labels: bool,
) -> None:
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center", transform=ax.transAxes)
        return

    pos = _layout(graph, layout=layout, seed=seed)
    colors = _node_colors(graph, node_label_attr)
    node_size = max(80, min(280, int(1800 / max(graph.number_of_nodes(), 1))))
    width = 1.0 if graph.number_of_edges() <= 50 else 0.6
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#8a8f98", width=width, alpha=0.75)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        cmap="tab20",
        node_size=node_size,
        linewidths=0.5,
        edgecolors="#222222",
    )
    if show_node_labels:
        labels = {n: str(graph.nodes[n].get(node_label_attr, n)) for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
    if show_edge_labels:
        edge_labels = {(u, v): str(data.get(edge_label_attr, "")) for u, v, data in graph.edges(data=True)}
        edge_labels = {edge: label for edge, label in edge_labels.items() if label}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw generated graphs for a trained model.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--samples", type=str, default=None, help="Optional generated sample pickle path.")
    parser.add_argument("--num-graphs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None, help="Randomly select graphs with this seed. By default, uses --start-index order.")
    parser.add_argument("--start-index", type=int, default=0, help="First graph index to draw when --seed is not set.")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--layout", choices=["spring", "kamada_kawai", "spectral", "circular"], default="spring")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--show-node-labels", action="store_true")
    parser.add_argument("--show-edge-labels", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    sample_file = Path(args.samples) if args.samples else sample_path(args.dataset, args.model)
    if not sample_file.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_file}. Run scripts/generate_samples.py first.")
    graphs = load_pickle(sample_file)
    if not isinstance(graphs, list) or not all(isinstance(g, nx.Graph) for g in graphs):
        raise TypeError(f"Expected a list of NetworkX graphs in {sample_file}.")

    selected = _select_graphs(graphs, num_graphs=args.num_graphs, seed=args.seed, start_index=args.start_index)
    cols = max(1, int(args.columns))
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for panel_idx, (graph_idx, graph) in enumerate(selected):
        ax = axes[panel_idx // cols][panel_idx % cols]
        title = f"#{graph_idx}  n={graph.number_of_nodes()} e={graph.number_of_edges()}"
        _draw_one(
            ax,
            graph,
            title=title,
            layout=args.layout,
            seed=int(args.seed if args.seed is not None else 42) + panel_idx,
            node_label_attr=args.node_label_attr,
            edge_label_attr=args.edge_label_attr,
            show_node_labels=args.show_node_labels,
            show_edge_labels=args.show_edge_labels,
        )

    fig.suptitle(f"{args.dataset}/{args.model} generated graphs", fontsize=12)
    fig.tight_layout()
    out = Path(args.output) if args.output else Path("outputs/figures/generated") / args.dataset / f"{args.model}_graphs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %d graph drawings to %s", len(selected), out)


if __name__ == "__main__":
    main()
