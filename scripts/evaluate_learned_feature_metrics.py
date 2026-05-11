from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

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
from empirical_comparison.metrics.learned_feature.distance import feature_mmd
from empirical_comparison.metrics.learned_feature.encoder import RandomGINPlaceholder, StructuralFeatureEncoder
from empirical_comparison.graphs.attributes import attribute_coverage, attribute_descriptor_features, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_pickle, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "learned_feature_metrics.json"


def _load_reference_graphs(dataset: str, dataset_root: str, reference_split: str) -> list[nx.Graph]:
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if reference_split not in splits:
        raise KeyError(f"Unknown reference split {reference_split!r}; available splits: {sorted(splits)}")
    graphs = list(splits[reference_split])
    if not graphs:
        raise ValueError(f"{reference_split!r} split for dataset '{dataset}' is empty.")
    return graphs


def _load_generated_graphs(dataset: str, model: str) -> list[nx.Graph]:
    sample_file = existing_sample_path(dataset, model)
    if not sample_file.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_file}. Run generate_samples.py first.")
    graphs = load_pickle(sample_file)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs as list, got {type(graphs)}.")
    if not graphs:
        raise ValueError(f"No generated graphs found in {sample_file}.")
    return graphs


def _maybe_subsample(graphs: Sequence[nx.Graph], max_graphs: int | None, seed: int) -> list[nx.Graph]:
    graphs = list(graphs)
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def _encode_graphs(graphs: Sequence[nx.Graph], encoder) -> np.ndarray:
    feats = np.asarray([encoder.encode(g) for g in graphs], dtype=np.float64)
    if feats.ndim != 2 or not np.all(np.isfinite(feats)):
        raise ValueError(f"Invalid feature matrix shape/content: {feats.shape}")
    return feats


def _feature_summary(x: np.ndarray) -> dict:
    return {
        "shape": list(x.shape),
        "mean_norm": float(np.linalg.norm(x, axis=1).mean()),
        "std_norm": float(np.linalg.norm(x, axis=1).std(ddof=0)),
        "mean_abs_value": float(np.abs(x).mean()),
    }


def _encoder_from_args(args):
    if args.encoder == "random_gin":
        encoder = RandomGINPlaceholder(feature_dim=args.feature_dim, seed=args.seed, normalize_output=True)
        encoder_note = "Fixed random-feature encoder; retained for backward compatibility and not a trained GNN."
        metric_name = "random_feature_mmd"
    else:
        encoder = StructuralFeatureEncoder(normalize_output=True)
        encoder_note = "Deterministic structural feature encoder; stronger fallback than random features but not a trained GNN."
        metric_name = "learned_feature_mmd"
    return encoder, encoder_note, metric_name


def _evaluate(args, *, seed: int, output_path: Path | None) -> dict:
    start = time.perf_counter()
    ref_graphs = _maybe_subsample(
        _load_reference_graphs(args.dataset, args.dataset_root, args.reference_split), args.max_graphs, args.seed
    )
    gen_graphs = _maybe_subsample(_load_generated_graphs(args.dataset, args.model), args.max_graphs, seed + 1)
    logger.info(
        "Evaluating feature MMD: dataset=%s model=%s encoder=%s reference_split=%s ref=%d gen=%d",
        args.dataset,
        args.model,
        args.encoder,
        args.reference_split,
        len(ref_graphs),
        len(gen_graphs),
    )

    encoder, encoder_note, metric_name = _encoder_from_args(args)
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
    ref_feats = _encode_graphs(ref_graphs, encoder)
    gen_feats = _encode_graphs(gen_graphs, encoder)
    attribute_features_used = False
    if not args.no_attribute_features:
        # Use shared vocab/dimensions for reference+generated graphs so the
        # attribute component is a valid two-sample feature space.
        attr_stats = fit_attribute_statistics(list(ref_graphs) + list(gen_graphs), attr_schema)
        if attr_stats.has_any_attributes:
            combined_attr = attribute_descriptor_features(
                list(ref_graphs) + list(gen_graphs),
                attr_schema,
                node_label_values=attr_stats.node_label_values,
                edge_label_values=attr_stats.edge_label_values,
                graph_label_values=attr_stats.graph_label_values,
                node_feature_dim=attr_stats.node_feature_dim,
                edge_feature_dim=attr_stats.edge_feature_dim,
                include_continuous=True,
            )
            ref_attr = combined_attr[: len(ref_graphs)]
            gen_attr = combined_attr[len(ref_graphs) :]
            ref_feats = np.concatenate([ref_feats, ref_attr], axis=1)
            gen_feats = np.concatenate([gen_feats, gen_attr], axis=1)
            ref_feats = ref_feats / np.maximum(np.linalg.norm(ref_feats, axis=1, keepdims=True), 1e-12)
            gen_feats = gen_feats / np.maximum(np.linalg.norm(gen_feats, axis=1, keepdims=True), 1e-12)
            attribute_features_used = True
    score = feature_mmd(ref_feats, gen_feats, sigma=args.sigma)
    elapsed = time.perf_counter() - start

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "learned_feature",
        "runtime_seconds": elapsed,
        "encoder": {
            "name": getattr(encoder, "name", encoder.__class__.__name__),
            "requested_encoder": args.encoder,
            "feature_dim": int(ref_feats.shape[1]),
            "seed": seed,
            "normalize_output": True,
            "note": encoder_note,
            "attribute_features_used": attribute_features_used,
        },
        "num_reference_graphs": len(ref_graphs),
        "num_generated_graphs": len(gen_graphs),
        "kernel": {"type": "rbf", "sigma": args.sigma, "sigma_note": "None means median heuristic was used inside feature_mmd."},
        "protocol": {
            "seed": seed,
            "reference_split": args.reference_split,
            "reference_split_note": "Default is train: this computes MMD between generated graph representations and training-data graph representations.",
            "max_graphs": args.max_graphs,
        },
        "graph_attributes": {
            "schema": attr_schema,
            "reference_attribute_coverage": attribute_coverage(ref_graphs, attr_schema),
            "generated_attribute_coverage": attribute_coverage(gen_graphs, attr_schema),
        },
        "debug": {"reference_features": _feature_summary(ref_feats), "generated_features": _feature_summary(gen_feats)},
        "results": {metric_name: float(score), "learned_feature_mmd": float(score)},
    }
    if output_path is None:
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME)
    save_json(payload, output_path)
    logger.info("Saved feature metrics to %s in %.2fs. learned_feature_mmd=%.8f", output_path, elapsed, score)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate feature-space MMD for generated graphs.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="train", help="Reference split for feature MMD. Default: train.")
    parser.add_argument("--encoder", choices=["structural", "random_gin"], default="structural")
    parser.add_argument("--no-attribute-features", action="store_true", help="Disable attribute-aware feature components in the fallback encoder.")
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--node-feature-attr", type=str, default="feats")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--edge-feature-attr", type=str, default="edge_attr")
    parser.add_argument("--graph-label-attr", type=str, default="graph_label")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    _evaluate(args, seed=args.seed, output_path=output_path)


if __name__ == "__main__":
    main()
