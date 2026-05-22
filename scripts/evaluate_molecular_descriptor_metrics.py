from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits, metadata_path
from empirical_comparison.evaluation.run_utils import evaluate_repeated_runs, existing_sample_path, metric_path
from empirical_comparison.generation.validity import quality_metrics
from empirical_comparison.graphs.attributes import (
    attribute_coverage,
    attribute_descriptor_features,
    canonicalize_graph_attributes,
    fit_attribute_statistics,
    normalize_schema,
)
from empirical_comparison.metrics.descriptor.descriptors import (
    clustering_histogram,
    degree_histogram,
    orbit_count_vector,
    spectral_histogram,
    structural_summary,
)
from empirical_comparison.metrics.descriptor.mmd import mmd_gaussian_emd, mmd_unbiased
from empirical_comparison.metrics.molecular.rdkit_validity import molecular_quality_metrics
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_json, load_pickle, load_yaml, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "molecular_descriptor_metrics.json"


def _load_reference_graphs(dataset: str, dataset_root: str, reference_split: str) -> list[nx.Graph]:
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if reference_split not in splits:
        raise KeyError(f"Unknown reference split {reference_split!r}; available splits: {sorted(splits)}")
    graphs = list(splits[reference_split])
    if not graphs:
        raise ValueError(f"{reference_split!r} split for dataset '{dataset}' is empty.")
    return graphs


def _load_train_graphs(dataset: str, dataset_root: str) -> list[nx.Graph]:
    return list(load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)["train"])


def _load_generated_graphs(dataset: str, model: str, run_id: int | None = None) -> list[nx.Graph]:
    sample_file = existing_sample_path(dataset, model, run_id=run_id)
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
    return [graphs[int(i)] for i in idx]


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
    return float(base), float(np.std(vals, ddof=0))


def _label_histogram_matrix(graphs: Sequence[nx.Graph], *, schema: dict, scope: str, values: Sequence[str]) -> np.ndarray:
    values = [str(v) for v in values]
    if not values:
        return np.zeros((len(graphs), 1), dtype=np.float64)
    index = {v: i for i, v in enumerate(values)}
    attr = schema["node_label_attr"] if scope == "node" else schema["edge_label_attr"]
    rows = np.zeros((len(graphs), len(values)), dtype=np.float64)
    for row_idx, graph in enumerate(graphs):
        items = (data for _, data in graph.nodes(data=True)) if scope == "node" else (data for _, _, data in graph.edges(data=True))
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


def _raw_attribute_values(dataset: str, dataset_root: str) -> tuple[list[str], list[str]]:
    try:
        meta = load_json(metadata_path(dataset, dataset_root))
        raw_stats = ((meta.get("graph_attributes") or {}).get("all_attribute_stats_raw") or {})
        node_values = [str(v) for v in raw_stats.get("node_label_values", [])]
        edge_values = [str(v) for v in raw_stats.get("edge_label_values", [])]
        return node_values, edge_values
    except Exception:
        return [], []


def _explicit_rdkit_node_mapping(dataset: str) -> tuple[list[str], str | None]:
    """Return optional user-supplied node-class -> atomic-number/symbol mapping.

    PyG ZINC atom_type ids are categorical.  Users can provide
    configs/datasets/zinc.yaml:rdkit_atomic_number_mapping as either a list
    indexed by node_label or a dict whose keys are node_label ids. Numeric
    values are encoded as "atomic_number=<n>" so rdkit_validity.py knows they
    are an explicit mapping rather than bare category ids.
    """
    cfg_path = ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if not cfg_path.exists():
        return [], None
    try:
        cfg = load_yaml(cfg_path) or {}
    except Exception:
        return [], None
    mapping = cfg.get("rdkit_atomic_number_mapping")
    if mapping is None:
        return [], None

    def encode(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)) and float(value).is_integer():
            return f"atomic_number={int(value)}"
        return str(value)

    if isinstance(mapping, dict):
        parsed: dict[int, str] = {}
        for key, value in mapping.items():
            try:
                parsed[int(key)] = encode(value)
            except Exception:
                continue
        if not parsed:
            return [], None
        values = [""] * (max(parsed) + 1)
        for key, value in parsed.items():
            values[key] = value
        return values, "configs/datasets/%s.yaml:rdkit_atomic_number_mapping" % dataset
    if isinstance(mapping, list):
        return [encode(v) for v in mapping], "configs/datasets/%s.yaml:rdkit_atomic_number_mapping" % dataset
    return [], None


def _evaluate(args, *, seed: int, output_path: Path | None) -> dict:
    start = time.perf_counter()
    max_ref_graphs = args.max_reference_graphs if args.max_reference_graphs is not None else args.max_graphs
    max_gen_graphs = args.max_generated_graphs if args.max_generated_graphs is not None else args.max_graphs
    ref_graphs = _subsample(_load_reference_graphs(args.dataset, args.dataset_root, args.reference_split), max_ref_graphs, seed)
    gen_graphs = _subsample(_load_generated_graphs(args.dataset, args.model, run_id=args.run_id), max_gen_graphs, seed + 1)
    train_graphs = _load_train_graphs(args.dataset, args.dataset_root)

    attr_schema = normalize_schema({
        "graph_attributes": {
            "enabled": args.attribute_schema_enabled,
            "node_label_attr": args.node_label_attr,
            "node_feature_attr": args.node_feature_attr,
            "edge_label_attr": args.edge_label_attr,
            "edge_feature_attr": args.edge_feature_attr,
            "graph_label_attr": args.graph_label_attr,
        }
    })
    attr_stats = fit_attribute_statistics(list(ref_graphs) + list(gen_graphs), attr_schema)
    ref_graphs, _ = canonicalize_graph_attributes(ref_graphs, attr_schema, attr_stats)
    gen_graphs, _ = canonicalize_graph_attributes(gen_graphs, attr_schema, attr_stats)
    raw_node_values, raw_edge_values = _raw_attribute_values(args.dataset, args.dataset_root)
    explicit_raw_node_values, rdkit_mapping_source = _explicit_rdkit_node_mapping(args.dataset)

    logger.info(
        "Evaluating molecular descriptor metrics: dataset=%s model=%s run_id=%s ref_split=%s ref=%d gen=%d",
        args.dataset,
        args.model,
        args.run_id,
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

    results: dict[str, Any] = {}
    debug: dict[str, dict] = {}
    for name, (fn, metric_kind) in descriptor_specs.items():
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

    generic_quality = quality_metrics(gen_graphs, reference_graphs=train_graphs, dataset=args.dataset)
    molecular_quality = molecular_quality_metrics(
        gen_graphs,
        train_graphs,
        node_label_attr=attr_schema["node_label_attr"],
        edge_label_attr=attr_schema["edge_label_attr"],
        raw_node_label_values=explicit_raw_node_values or raw_node_values or attr_stats.node_label_values,
        raw_edge_label_values=raw_edge_values or attr_stats.edge_label_values,
        dataset=args.dataset,
    )
    results.update(generic_quality)
    results.update(molecular_quality)

    elapsed = time.perf_counter() - start
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": args.run_id,
        "metric_family": "molecular_descriptor_based",
        "num_reference_graphs": len(ref_graphs),
        "num_generated_graphs": len(gen_graphs),
        "runtime_seconds": elapsed,
        "protocol": {
            "seed": seed,
            "run_id": args.run_id,
            "reference_split": args.reference_split,
            "max_graphs": args.max_graphs,
            "max_reference_graphs": max_ref_graphs,
            "max_generated_graphs": max_gen_graphs,
            "sigma": args.sigma,
            "num_bootstrap": args.num_bootstrap,
            "degree_bins": args.degree_bins,
            "clustering_bins": args.clustering_bins,
            "spectral_bins": args.spectral_bins,
            "max_degree": args.max_degree,
            "orca_exec": args.orca_exec,
            "skip_orbit": args.skip_orbit,
            "orbit_log_transform": args.orbit_log_transform,
            "attribute_mmd": not args.no_attribute_mmd,
            "rdkit_validity": True,
            "attribute_schema": attr_schema,
            "raw_node_label_values_for_rdkit": explicit_raw_node_values or raw_node_values,
            "rdkit_atomic_number_mapping_source": rdkit_mapping_source,
            "raw_edge_label_values_for_rdkit": raw_edge_values,
        },
        "notes": {
            "rdkit_validity": "Generated graphs are converted to RDKit molecules using node_label/edge_type mappings and sanitized. When RDKit construction fails, a conservative valence fallback is recorded by validity_backend.",
            "uniqueness_rate": "Unique canonical RDKit SMILES among valid generated molecules, divided by valid generated molecules.",
            "novelty_rate": "Novel unique valid generated SMILES not present in the training split, divided by unique valid generated SMILES.",
            "valid_unique_novel_rate": "Novel unique valid generated molecules divided by all generated graphs.",
            "atom_type_mmd": "RBF MMD over per-graph atom-label histograms.",
            "bond_type_mmd": "RBF MMD over per-graph bond-label histograms.",
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
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME, run_id=args.run_id)
    save_json(payload, output_path)
    logger.info("Saved molecular descriptor metrics to %s in %.2fs", output_path, elapsed)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate molecular graph descriptor and RDKit validity metrics.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None, help="Optional repeated-run id used to load run-specific samples and write run-specific metrics.")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Evaluate multiple run-specific sample files and write an aggregate metric JSON with across-run means.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Backward-compatible cap applied to both reference and generated graphs unless side-specific caps are supplied.")
    parser.add_argument("--max-reference-graphs", type=int, default=None)
    parser.add_argument("--max-generated-graphs", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--num-bootstrap", type=int, default=0)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-bins", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=100)
    parser.add_argument("--orca-exec", type=str, default=None)
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--orbit-log-transform", action="store_true")
    parser.add_argument("--no-attribute-mmd", action="store_true")
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--node-feature-attr", type=str, default="feats")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--edge-feature-attr", type=str, default="edge_attr")
    parser.add_argument("--graph-label-attr", type=str, default="graph_label")
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    evaluate_repeated_runs(args, metric_filename=METRIC_FILENAME, evaluate_fn=_evaluate, base_seed=args.seed, output_path=output_path)


if __name__ == "__main__":
    main()
