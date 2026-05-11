from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

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

    results: dict[str, float] = {}
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

    results.update(quality_metrics(gen_graphs, reference_graphs=train_graphs, dataset=args.dataset))
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
            "attribute_schema": attr_schema,
        },
        "notes": {
            "orbit_mmd": "ORCA-based 4-node orbit-count MMD when ORCA is configured; otherwise skipped.",
            "structural_summary_mmd": "Lightweight structural-summary fallback, not a graphlet/orbit metric.",
            "attribute_mmd": "RBF MMD over node/edge attribute histograms and continuous attribute moments when attributes are present.",
            "classifier_auc": "For classifier metrics, values near 0.5 indicate low real/generated separability.",
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
