from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits
from empirical_comparison.graphs.attributes import attribute_coverage, fit_attribute_statistics, normalize_schema
from empirical_comparison.registry import available_datasets
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _stats(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    graphs = list(graphs)
    if not graphs:
        return {
            "num_graphs": 0,
            "num_nodes_min": 0,
            "num_nodes_mean": 0.0,
            "num_nodes_max": 0,
            "num_edges_min": 0,
            "num_edges_mean": 0.0,
            "num_edges_max": 0,
            "density_mean": 0.0,
            "avg_degree_mean": 0.0,
            "connected_rate": 0.0,
            "self_loop_graph_rate": 0.0,
        }

    nodes = np.asarray([g.number_of_nodes() for g in graphs], dtype=np.float64)
    edges = np.asarray([g.number_of_edges() for g in graphs], dtype=np.float64)
    densities = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs], dtype=np.float64)
    avg_degrees = np.asarray([
        (2.0 * g.number_of_edges() / g.number_of_nodes()) if g.number_of_nodes() > 0 else 0.0
        for g in graphs
    ], dtype=np.float64)
    connected = []
    self_loop_graphs = []
    for graph in graphs:
        try:
            connected.append(graph.number_of_nodes() > 0 and nx.is_connected(graph))
        except Exception:
            connected.append(False)
        try:
            self_loop_graphs.append(len(list(nx.selfloop_edges(graph))) > 0)
        except Exception:
            self_loop_graphs.append(False)

    return {
        "num_graphs": int(len(graphs)),
        "num_nodes_min": int(nodes.min()),
        "num_nodes_mean": float(nodes.mean()),
        "num_nodes_max": int(nodes.max()),
        "num_edges_min": int(edges.min()),
        "num_edges_mean": float(edges.mean()),
        "num_edges_max": int(edges.max()),
        "density_mean": float(densities.mean()),
        "avg_degree_mean": float(avg_degrees.mean()),
        "connected_rate": _rate(connected),
        "self_loop_graph_rate": _rate(self_loop_graphs),
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("split", "Split"),
        ("num_graphs", "Graphs"),
        ("num_nodes_min", "Nodes min"),
        ("num_nodes_mean", "Nodes mean"),
        ("num_nodes_max", "Nodes max"),
        ("num_edges_min", "Edges min"),
        ("num_edges_mean", "Edges mean"),
        ("num_edges_max", "Edges max"),
        ("density_mean", "Density mean"),
        ("avg_degree_mean", "Avg degree"),
        ("connected_rate", "Connected"),
        ("self_loop_graph_rate", "Self-loop graphs"),
    ]
    rendered = []
    for row in rows:
        rendered.append([_format_value(row.get(key, "")) for key, _ in columns])
    widths = [
        max(len(title), *(len(row[i]) for row in rendered))
        for i, (_, title) in enumerate(columns)
    ]
    header = "  ".join(title.ljust(widths[i]) for i, (_, title) in enumerate(columns))
    sep = "  ".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in rendered:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


def _attribute_summary(splits: dict[str, list[nx.Graph]], attr_schema: dict[str, Any]) -> dict[str, Any]:
    all_graphs = [graph for graphs in splits.values() for graph in graphs]
    attr_stats = fit_attribute_statistics(all_graphs, attr_schema)
    return {
        "node_label_values": list(attr_stats.node_label_values),
        "edge_label_values": list(attr_stats.edge_label_values),
        "node_feature_dim": attr_stats.node_feature_dim,
        "edge_feature_dim": attr_stats.edge_feature_dim,
        "graph_label_values": list(attr_stats.graph_label_values),
        "coverage": {split: attribute_coverage(graphs, attr_schema) for split, graphs in splits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print basic statistics for a prepared benchmark dataset.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Dataset splits to include.")
    parser.add_argument("--build-if-missing", action="store_true", help="Build dataset splits from config if persisted files are missing.")
    parser.add_argument("--config", type=str, default=None, help="Optional dataset config used only with --build-if-missing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text table.")
    parser.add_argument("--include-attributes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--node-feature-attr", type=str, default="feats")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--edge-feature-attr", type=str, default="edge_attr")
    parser.add_argument("--graph-label-attr", type=str, default="graph_label")
    args = parser.parse_args()

    splits = load_dataset_splits(
        args.dataset,
        output_root=args.dataset_root,
        build_if_missing=args.build_if_missing,
        config_path=args.config,
    )
    selected = {split: list(splits[split]) for split in args.splits if split in splits}
    if not selected:
        raise ValueError(f"No requested splits found. Requested={args.splits}; available={sorted(splits)}")

    rows = []
    for split, graphs in selected.items():
        row = {"split": split}
        row.update(_stats(graphs))
        rows.append(row)
    if len(selected) > 1:
        all_graphs = [graph for graphs in selected.values() for graph in graphs]
        row = {"split": "all"}
        row.update(_stats(all_graphs))
        rows.append(row)

    attr_schema = normalize_schema({"graph_attributes": {
        "enabled": args.attribute_schema_enabled,
        "node_label_attr": args.node_label_attr,
        "node_feature_attr": args.node_feature_attr,
        "edge_label_attr": args.edge_label_attr,
        "edge_feature_attr": args.edge_feature_attr,
        "graph_label_attr": args.graph_label_attr,
    }})
    output: dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "splits": rows,
    }
    if args.include_attributes:
        output["graph_attributes"] = _attribute_summary(selected, attr_schema)

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return

    print(f"Dataset: {args.dataset}")
    _print_table(rows)
    if args.include_attributes:
        attrs = output["graph_attributes"]
        print()
        print("Attributes:")
        print(f"  node_label_values: {len(attrs['node_label_values'])}")
        print(f"  edge_label_values: {len(attrs['edge_label_values'])}")
        print(f"  node_feature_dim: {_safe_float(attrs['node_feature_dim']):.0f}")
        print(f"  edge_feature_dim: {_safe_float(attrs['edge_feature_dim']):.0f}")
        print(f"  graph_label_values: {len(attrs['graph_label_values'])}")


if __name__ == "__main__":
    main()
