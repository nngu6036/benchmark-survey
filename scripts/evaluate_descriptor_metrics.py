from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits
from empirical_comparison.evaluation.run_utils import (
    existing_sample_path,
    metric_path,
)
from empirical_comparison.generation.validity import quality_metrics
from empirical_comparison.graphs.attributes import attribute_coverage, attribute_descriptor_features, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.metrics.descriptor.descriptors import (
    clustering_histogram,
    degree_histogram,
    orbit_count_vector,
    spectral_histogram,
    structural_summary,
)
from empirical_comparison.metrics.descriptor.mmd import mmd_gaussian_emd, mmd_unbiased
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_pickle, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "descriptor_metrics.json"


def _load_reference_graphs(dataset: str, dataset_root: str, reference_split: str) -> list[nx.Graph]:
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if reference_split not in splits:
        raise KeyError(f"Unknown reference split {reference_split!r}; available splits: {sorted(splits)}")
    graphs = list(splits[reference_split])
    if not graphs:
        raise ValueError(f"{reference_split!r} split for dataset '{dataset}' is empty.")
    return graphs


def _load_train_graphs(dataset: str, dataset_root: str) -> list[nx.Graph]:
    try:
        return list(load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)["train"])
    except Exception:
        return []


def _load_generated_graphs(dataset: str, model: str) -> list[nx.Graph]:
    sample_file = existing_sample_path(dataset, model)
    if not sample_file.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_file}. Run generate_samples.py first.")
    graphs = load_pickle(sample_file)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs to be a list, got {type(graphs)}")
    if not graphs:
        raise ValueError(f"No generated graphs found in {sample_file}.")
    return graphs


def _subsample(graphs: Sequence[nx.Graph], max_graphs: int | None, seed: int) -> list[nx.Graph]:
    graphs = list(graphs)
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def _descriptor_matrix(graphs: Sequence[nx.Graph], fn: Callable[[nx.Graph], np.ndarray]) -> np.ndarray:
    x = np.asarray([fn(g) for g in graphs], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Descriptor function returned invalid shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Descriptor matrix contains NaN or Inf values.")
    return x


def _label_histogram_matrix(graphs: Sequence[nx.Graph], *, schema: dict, scope: str, values: Sequence[str]) -> np.ndarray:
    values = [str(v) for v in values]
    if not values:
        return np.zeros((len(graphs), 1), dtype=np.float64)
    index = {v: i for i, v in enumerate(values)}
    attr = schema["node_label_attr"] if scope == "node" else schema["edge_label_attr"]
    rows = np.zeros((len(graphs), len(values)), dtype=np.float64)
    for row_idx, graph in enumerate(graphs):
        if scope == "node":
            items = (data for _, data in graph.nodes(data=True))
        else:
            items = (data for _, _, data in graph.edges(data=True))
        for data in items:
            key = str(data.get(attr, ""))
            if key in index:
                rows[row_idx, index[key]] += 1.0
        total = rows[row_idx].sum()
        if total > 0:
            rows[row_idx] /= total
    return rows


def _compute_label_mmd(
    *,
    name: str,
    ref_graphs: Sequence[nx.Graph],
    gen_graphs: Sequence[nx.Graph],
    attr_schema: dict,
    scope: str,
    values: Sequence[str],
    sigma: float | None,
    num_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    logger.info("Computing %s", name)
    ref_desc = _label_histogram_matrix(ref_graphs, schema=attr_schema, scope=scope, values=values)
    gen_desc = _label_histogram_matrix(gen_graphs, schema=attr_schema, scope=scope, values=values)
    mean, std = _mmd_with_optional_bootstrap(ref_desc, gen_desc, metric_kind="rbf", sigma=sigma, num_bootstrap=num_bootstrap, seed=seed)
    results = {name: mean}
    if std is not None:
        results[f"{name}_bootstrap_std"] = std
    debug = {
        "reference_shape": list(ref_desc.shape),
        "generated_shape": list(gen_desc.shape),
        "source_attribute": attr_schema["node_label_attr"] if scope == "node" else attr_schema["edge_label_attr"],
        "mmd_kernel": "rbf",
    }
    return results, debug


def _graph_label_fingerprint(graph: nx.Graph, schema: dict) -> str:
    node_attr = schema["node_label_attr"]
    edge_attr = schema["edge_label_attr"]
    try:
        h = nx.Graph()
        for node, data in graph.nodes(data=True):
            h.add_node(node, **{node_attr: str(data.get(node_attr, ""))})
        for u, v, data in graph.edges(data=True):
            h.add_edge(u, v, **{edge_attr: str(data.get(edge_attr, ""))})
        return nx.weisfeiler_lehman_graph_hash(h, node_attr=node_attr, edge_attr=edge_attr)
    except Exception:
        nodes = sorted(str(data.get(node_attr, "")) for _, data in graph.nodes(data=True))
        edges = sorted(
            (str(graph.nodes[u].get(node_attr, "")), str(graph.nodes[v].get(node_attr, "")), str(data.get(edge_attr, "")))
            for u, v, data in graph.edges(data=True)
        )
        return repr((nodes, edges))


def _as_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _qm9_atomic_number(label: Any, attr_stats) -> int | None:
    label_int = _as_int(label)
    values = list(getattr(attr_stats, "node_label_values", []) or [])
    if label_int is not None and 0 <= label_int < len(values):
        mapped = _as_int(values[label_int])
        if mapped in {1, 6, 7, 8, 9}:
            return mapped
    if label_int in {1, 6, 7, 8, 9}:
        return label_int
    # Common QM9 class order used by molecular diffusion code with hydrogens.
    class_order = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
    return class_order.get(label_int)


def _qm9_bond_order(edge_type: Any) -> float | None:
    value = _as_int(edge_type)
    if value is None:
        return None
    if value == 1:
        return 1.0
    if value == 2:
        return 2.0
    if value == 3:
        return 3.0
    if value == 4:
        return 1.5
    return None


def _is_qm9_valid_graph(graph: nx.Graph, attr_schema: dict, attr_stats) -> bool:
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return False
    if any(u == v for u, v in graph.edges()):
        return False
    node_attr = attr_schema["node_label_attr"]
    edge_attr = attr_schema["edge_label_attr"]
    max_valence = {1: 1.0, 6: 4.0, 7: 3.0, 8: 2.0, 9: 1.0}
    valence: Counter[Any] = Counter()
    for node, data in graph.nodes(data=True):
        atomic_number = _qm9_atomic_number(data.get(node_attr), attr_stats)
        if atomic_number not in max_valence:
            return False
        valence[node] = 0.0
    for u, v, data in graph.edges(data=True):
        order = _qm9_bond_order(data.get(edge_attr, 1))
        if order is None:
            return False
        valence[u] += order
        valence[v] += order
    for node, data in graph.nodes(data=True):
        atomic_number = _qm9_atomic_number(data.get(node_attr), attr_stats)
        if valence[node] > max_valence[atomic_number] + 1e-9:
            return False
    return True


def _qm9_quality_metrics(gen_graphs: Sequence[nx.Graph], train_graphs: Sequence[nx.Graph], attr_schema: dict, attr_stats) -> dict[str, float | None]:
    if not gen_graphs:
        return {
            "validity_rate": 0.0,
            "dataset_validity_rate": 0.0,
            "uniqueness_rate": 0.0,
            "novelty_rate": 0.0 if train_graphs else None,
        }
    valid_flags = [_is_qm9_valid_graph(g, attr_schema, attr_stats) for g in gen_graphs]
    valid_graphs = [g for g, ok in zip(gen_graphs, valid_flags) if ok]
    fingerprints = [_graph_label_fingerprint(g, attr_schema) for g in valid_graphs]
    train_fingerprints = {_graph_label_fingerprint(g, attr_schema) for g in train_graphs}
    novelty = None
    if train_graphs:
        novelty = float(np.mean([fp not in train_fingerprints for fp in fingerprints])) if fingerprints else 0.0
    uniqueness = float(len(set(fingerprints)) / len(fingerprints)) if fingerprints else 0.0
    validity = float(np.mean(valid_flags))
    return {
        "validity_rate": validity,
        "dataset_validity_rate": validity,
        "uniqueness_rate": uniqueness,
        "novelty_rate": novelty,
    }


def _mmd_with_optional_bootstrap(ref_desc, gen_desc, *, metric_kind: str, sigma: float | None, num_bootstrap: int, seed: int):
    if metric_kind == "rbf":
        compute = lambda a, b: mmd_unbiased(a, b, sigma=sigma)
    elif metric_kind == "emd":
        sigma_value = 1.0 if sigma is None else sigma
        compute = lambda a, b: mmd_gaussian_emd(a, b, sigma=sigma_value, unbiased=False)
    else:
        raise ValueError(f"Unknown metric_kind: {metric_kind}")
    base = compute(ref_desc, gen_desc)
    if num_bootstrap <= 0:
        return float(base), None
    rng = np.random.default_rng(seed)
    n = min(len(ref_desc), len(gen_desc))
    vals = []
    for _ in range(num_bootstrap):
        ref_idx = rng.choice(len(ref_desc), size=n, replace=True)
        gen_idx = rng.choice(len(gen_desc), size=n, replace=True)
        vals.append(compute(ref_desc[ref_idx], gen_desc[gen_idx]))
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def _evaluate(args, *, seed: int, output_path: Path | None) -> dict:
    start = time.perf_counter()
    ref_graphs = _subsample(_load_reference_graphs(args.dataset, args.dataset_root, args.reference_split), args.max_graphs, args.seed)
    gen_graphs = _subsample(_load_generated_graphs(args.dataset, args.model), args.max_graphs, seed + 1)
    train_graphs = _load_train_graphs(args.dataset, args.dataset_root)
    attr_schema = normalize_schema({"graph_attributes": {
        "enabled": args.attribute_schema_enabled,
        "node_label_attr": args.node_label_attr,
        "node_feature_attr": args.node_feature_attr,
        "edge_label_attr": args.edge_label_attr,
        "edge_feature_attr": args.edge_feature_attr,
        "graph_label_attr": args.graph_label_attr,
    }})
    attr_stats = fit_attribute_statistics(list(ref_graphs) + list(gen_graphs), attr_schema)
    ref_graphs, _ = canonicalize_graph_attributes(ref_graphs, attr_schema, attr_stats)
    gen_graphs, _ = canonicalize_graph_attributes(gen_graphs, attr_schema, attr_stats)

    logger.info(
        "Evaluating descriptor metrics: dataset=%s model=%s reference_split=%s ref=%d gen=%d",
        args.dataset,
        args.model,
        args.reference_split,
        len(ref_graphs),
        len(gen_graphs),
    )

    descriptor_specs: dict[str, tuple[Callable[[nx.Graph], np.ndarray], str]] = {
        "degree_mmd": (lambda g: degree_histogram(g, bins=args.degree_bins, max_degree=args.max_degree), "emd"),
        "clustering_mmd": (lambda g: clustering_histogram(g, bins=args.clustering_bins), "emd"),
        "spectral_mmd": (lambda g: spectral_histogram(g, bins=args.spectral_bins), "emd"),
        "structural_summary_mmd": (structural_summary, "rbf"),
    }
    if not args.skip_orbit:
        descriptor_specs["orbit_mmd"] = (
            lambda g: orbit_count_vector(g, orca_exec=args.orca_exec, normalize=True, log_transform=args.orbit_log_transform),
            "emd",
        )
    if not args.no_attribute_mmd and attr_stats.has_any_attributes:
        descriptor_specs["attribute_mmd"] = (
            lambda g: attribute_descriptor_features(
                [g],
                attr_schema,
                node_label_values=attr_stats.node_label_values,
                edge_label_values=attr_stats.edge_label_values,
                graph_label_values=attr_stats.graph_label_values,
                node_feature_dim=attr_stats.node_feature_dim,
                edge_feature_dim=attr_stats.edge_feature_dim,
                include_continuous=True,
            )[0],
            "rbf",
        )

    results: dict[str, float | int | None] = {}
    debug: dict[str, dict] = {}
    for name, (fn, metric_kind) in descriptor_specs.items():
        logger.info("Computing %s", name)
        try:
            ref_desc = _descriptor_matrix(ref_graphs, fn)
            gen_desc = _descriptor_matrix(gen_graphs, fn)
        except FileNotFoundError:
            if name == "orbit_mmd":
                logger.warning("Skipping orbit_mmd because ORCA is unavailable. Use --orca-exec or set ORCA_EXEC to enable it.")
                continue
            raise
        mean, std = _mmd_with_optional_bootstrap(ref_desc, gen_desc, metric_kind=metric_kind, sigma=args.sigma, num_bootstrap=args.num_bootstrap, seed=seed)
        results[name] = mean
        if std is not None:
            results[f"{name}_bootstrap_std"] = std
        debug[name] = {
            "reference_shape": list(ref_desc.shape),
            "generated_shape": list(gen_desc.shape),
            "reference_mean_norm": float(np.linalg.norm(ref_desc, axis=1).mean()),
            "generated_mean_norm": float(np.linalg.norm(gen_desc, axis=1).mean()),
            "mmd_kernel": "gaussian_emd" if metric_kind == "emd" else "rbf",
        }

    if attr_stats.node_label_values:
        node_values = [str(i) for i in range(len(attr_stats.node_label_values))]
        metric, metric_debug = _compute_label_mmd(
            name="atom_type_mmd",
            ref_graphs=ref_graphs,
            gen_graphs=gen_graphs,
            attr_schema=attr_schema,
            scope="node",
            values=node_values,
            sigma=args.sigma,
            num_bootstrap=args.num_bootstrap,
            seed=seed,
        )
        results.update(metric)
        debug["atom_type_mmd"] = metric_debug

    if attr_stats.edge_label_values:
        edge_values = [str(i + 1) for i in range(len(attr_stats.edge_label_values))]
        metric, metric_debug = _compute_label_mmd(
            name="bond_type_mmd",
            ref_graphs=ref_graphs,
            gen_graphs=gen_graphs,
            attr_schema=attr_schema,
            scope="edge",
            values=edge_values,
            sigma=args.sigma,
            num_bootstrap=args.num_bootstrap,
            seed=seed,
        )
        results.update(metric)
        debug["bond_type_mmd"] = metric_debug

    quality = quality_metrics(gen_graphs, reference_graphs=train_graphs, dataset=args.dataset)
    if str(args.dataset).lower() == "qm9":
        quality.update(_qm9_quality_metrics(gen_graphs, train_graphs, attr_schema, attr_stats))
    results.update(quality)
    elapsed = time.perf_counter() - start
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "descriptor_based",
        "num_reference_graphs": len(ref_graphs),
        "num_generated_graphs": len(gen_graphs),
        "runtime_seconds": elapsed,
        "protocol": {
            "seed": seed,
            "reference_split": args.reference_split,
            "max_graphs": args.max_graphs,
            "sigma": args.sigma,
            "sigma_note": "For EMD-kernel MMD, None uses sigma=1.0. For RBF MMD, None uses median heuristic.",
            "num_bootstrap": args.num_bootstrap,
            "degree_bins": args.degree_bins,
            "clustering_bins": args.clustering_bins,
            "spectral_bins": args.spectral_bins,
            "max_degree": args.max_degree,
            "orca_exec": args.orca_exec,
            "skip_orbit": args.skip_orbit,
            "orbit_log_transform": args.orbit_log_transform,
            "attribute_mmd": not args.no_attribute_mmd,
            "qm9_molecular_validity": str(args.dataset).lower() == "qm9",
            "attribute_schema": attr_schema,
        },
        "notes": {
            "orbit_mmd": "ORCA-based 4-node orbit-count MMD when ORCA is configured; otherwise skipped.",
            "structural_summary_mmd": "Lightweight structural-summary fallback, not a graphlet/orbit metric.",
            "attribute_mmd": "RBF MMD over node/edge attribute histograms and continuous attribute moments when attributes are present.",
            "atom_type_mmd": "RBF MMD over per-graph atom-type histograms when node labels are present.",
            "bond_type_mmd": "RBF MMD over per-graph bond-type histograms when edge labels are present.",
            "qm9_validity": "For QM9, validity/uniqueness/novelty use a lightweight valence check over generated node labels and bond types.",
        },
        "graph_attributes": {
            "schema": attr_schema,
            "reference_attribute_coverage": attribute_coverage(ref_graphs, attr_schema),
            "generated_attribute_coverage": attribute_coverage(gen_graphs, attr_schema),
        },
        "debug": debug,
        "results": results,
    }
    if output_path is None:
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME)
    save_json(payload, output_path)
    logger.info("Saved descriptor metrics to %s in %.2fs", output_path, elapsed)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate descriptor-based graph generation metrics.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--num-bootstrap", type=int, default=0)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-bins", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=100)
    parser.add_argument("--orca-exec", type=str, default=None)
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--orbit-log-transform", action="store_true")
    parser.add_argument("--no-attribute-mmd", action="store_true", help="Disable attribute descriptor MMD.")
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--node-feature-attr", type=str, default="feats")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--edge-feature-attr", type=str, default="edge_attr")
    parser.add_argument("--graph-label-attr", type=str, default="graph_label")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    _evaluate(args, seed=args.seed, output_path=output_path)


if __name__ == "__main__":
    main()
