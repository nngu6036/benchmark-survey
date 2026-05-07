from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits
from empirical_comparison.evaluation.run_utils import (
    aggregate_metric_path,
    aggregate_numeric_results,
    existing_sample_path,
    explicit_run_selection,
    metric_path,
    parse_run_ids,
    run_seed,
    should_use_run_paths,
)
from empirical_comparison.graphs.attributes import attribute_coverage, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.metrics.classifier.pgs import DescriptorConfig, polygraphscore
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_pickle, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "classifier_metrics.json"
DEFAULT_DESCRIPTORS = ["degree", "clustering", "spectral", "orbit4", "orbit5", "gin", "attributes"]


def _load_reference_graphs(dataset: str, dataset_root: str, reference_split: str) -> list:
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if reference_split not in splits:
        raise KeyError(f"Unknown reference split {reference_split!r}; available splits: {sorted(splits)}")
    graphs = list(splits[reference_split])
    if len(graphs) == 0:
        raise ValueError(f"{reference_split!r} split for dataset {dataset!r} is empty.")
    return graphs


def _load_generated_graphs(dataset: str, model: str, run_id: int | None) -> list:
    sample_file = existing_sample_path(dataset, model, run_id)
    if not sample_file.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_file}. Run generate_samples.py first.")
    graphs = load_pickle(sample_file)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs as list, got {type(graphs)}.")
    if len(graphs) == 0:
        raise ValueError(f"No generated graphs found in {sample_file}.")
    return graphs


def _subsample(graphs: list, max_graphs: int | None, seed: int) -> list:
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return list(graphs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def _numeric_mean_std(split_payloads: list[dict]) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for payload in split_payloads:
        for key, value in (payload.get("results") or {}).items():
            if isinstance(value, (int, float, np.number)):
                values_by_key[key].append(float(value))
    out: dict[str, float] = {}
    for key, vals in values_by_key.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = float(arr.mean())
        out[f"{key}_split_mean"] = float(arr.mean())
        out[f"{key}_split_std"] = float(arr.std(ddof=0))
    return out


def _descriptor_summary(split_payloads: list[dict]) -> dict[str, dict[str, float]]:
    per_desc: dict[str, list[float]] = defaultdict(list)
    per_desc_cv: dict[str, list[float]] = defaultdict(list)
    for payload in split_payloads:
        for item in payload.get("descriptor_results", []):
            name = str(item.get("descriptor"))
            if isinstance(item.get("test_score"), (int, float, np.number)):
                per_desc[name].append(float(item["test_score"]))
            if isinstance(item.get("cv_score"), (int, float, np.number)):
                per_desc_cv[name].append(float(item["cv_score"]))
    summary: dict[str, dict[str, float]] = {}
    for name, vals in per_desc.items():
        arr = np.asarray(vals, dtype=np.float64)
        cv_arr = np.asarray(per_desc_cv.get(name, []), dtype=np.float64)
        summary[name] = {
            "test_mean": float(arr.mean()),
            "test_std": float(arr.std(ddof=0)),
            "cv_mean": float(cv_arr.mean()) if cv_arr.size else float("nan"),
            "cv_std": float(cv_arr.std(ddof=0)) if cv_arr.size else float("nan"),
            "num_partitions": int(arr.size),
        }
    return summary


def _evaluate_one_run(args, *, run_id: int, logical_run_id: int | None, seed: int, output_path: Path | None) -> dict:
    start = time.perf_counter()
    ref_graphs = _subsample(_load_reference_graphs(args.dataset, args.dataset_root, args.reference_split), args.max_graphs, seed)
    gen_graphs = _subsample(_load_generated_graphs(args.dataset, args.model, logical_run_id), args.max_graphs, seed + 1)

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

    descriptor_names = args.descriptors or list(DEFAULT_DESCRIPTORS)
    if args.skip_orbits or args.skip_orbit:
        descriptor_names = [d for d in descriptor_names if not str(d).lower().startswith(("orbit", "orb"))]
    if args.no_attribute_descriptor:
        descriptor_names = [d for d in descriptor_names if str(d).lower() not in {"attributes", "attribute", "attrs"}]

    logger.info(
        "Evaluating PGS: dataset=%s model=%s run_id=%s ref_split=%s ref=%d gen=%d partitions=%d classifier=%s mode=%s",
        args.dataset,
        args.model,
        "legacy" if logical_run_id is None else logical_run_id,
        args.reference_split,
        len(ref_graphs),
        len(gen_graphs),
        args.num_splits,
        args.classifier,
        args.mode,
    )

    descriptor_cfg = DescriptorConfig(
        degree_bins=args.degree_bins,
        clustering_bins=args.clustering_bins,
        spectral_bins=args.spectral_bins,
        max_degree=args.max_degree,
        orca_exec=args.orca_exec,
        graph_attributes=attr_schema,
        gin_dim=args.gin_dim,
        seed=seed,
    )

    split_payloads = []
    for split_id in range(max(1, int(args.num_splits))):
        split_seed = seed + split_id * 9973
        pgs_payload = polygraphscore(
            ref_graphs,
            gen_graphs,
            descriptor_names=descriptor_names,
            descriptor_config=descriptor_cfg,
            mode=args.mode,
            classifier=args.classifier,
            cv_folds=args.cv_folds,
            seed=split_seed,
            skip_unavailable=True,
            device=args.device,
        )
        pgs_payload["split_id"] = split_id
        pgs_payload["seed"] = split_seed
        split_payloads.append(pgs_payload)
        logger.info(
            "run=%s pgs_partition=%d pgs=%.4f selected=%s classifier=%s",
            "legacy" if logical_run_id is None else logical_run_id,
            split_id,
            float(pgs_payload["results"].get("polygraphscore", float("nan"))),
            pgs_payload.get("best_descriptor"),
            pgs_payload.get("classifier"),
        )

    results = _numeric_mean_std(split_payloads)
    # Compatibility aliases for the table/reporter.
    if "polygraphscore" in results:
        results["polygraphscore_mean"] = results["polygraphscore"]
        results["polygraphscore_std"] = results.get("polygraphscore_split_std", 0.0)
        results["pgs_mean"] = results["polygraphscore"]
        results["pgs_std"] = results.get("polygraphscore_split_std", 0.0)
    if "pgs_js_distance" in results:
        results["pgs_js_distance_mean"] = results["pgs_js_distance"]
        results["pgs_js_distance_std"] = results.get("pgs_js_distance_split_std", 0.0)

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": logical_run_id,
        "metric_family": "polygraphscore_classifier",
        "runtime_seconds": time.perf_counter() - start,
        "metric_name": "PolyGraphScore",
        "feature_representation": {
            "name": "descriptor_wise_polygraphscore",
            "descriptors_requested": descriptor_names,
            "degree_bins": args.degree_bins,
            "clustering_bins": args.clustering_bins,
            "spectral_bins": args.spectral_bins,
            "max_degree": args.max_degree,
            "gin_dim": args.gin_dim,
        },
        "classifier": {
            "requested": args.classifier,
            "resolved_per_partition": [p.get("classifier") for p in split_payloads],
            "note": "TabPFN is used when installed/requested; standardized logistic regression is the documented fallback/ablation.",
        },
        "protocol": {
            "num_reference_graphs": len(ref_graphs),
            "num_generated_graphs": len(gen_graphs),
            "reference_split": args.reference_split,
            "fit_test_split": "Each PGS partition randomly halves reference and generated graphs into fit/test sets; descriptor selection uses CV on the fit half; final PGS is evaluated on the held-out test half.",
            "descriptor_selection": "Select descriptor with maximum CV lower-bound score, then report its held-out test score.",
            "num_repeated_partitions": int(args.num_splits),
            "cv_folds_on_fit_set": args.cv_folds,
            "mode": args.mode,
            "seed": seed,
            "base_seed": args.seed,
            "run_id": logical_run_id,
        },
        "graph_attributes": {
            "schema": attr_schema,
            "reference_attribute_coverage": attribute_coverage(ref_graphs, attr_schema),
            "generated_attribute_coverage": attribute_coverage(gen_graphs, attr_schema),
        },
        "interpretation": {
            "polygraphscore": "Paper-style PGS: JS-distance lower-bound estimate in [0, 1]; lower is closer to the reference distribution.",
            "pgs_best_descriptor": "Descriptor selected by highest cross-validation score on the fit set for each partition.",
            "classifier_auc": "Diagnostic only; 0.5 is indistinguishable and 1.0 is highly separable.",
        },
        "results": results,
        "descriptor_summary": _descriptor_summary(split_payloads),
        "split_results": split_payloads,
    }
    if output_path is None:
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME, logical_run_id)
    save_json(payload, output_path)
    logger.info("Saved PGS metrics to %s", output_path)
    return payload


def _save_aggregate(args, run_ids: list[int], run_payloads: list[dict], output_path: Path | None) -> None:
    agg = aggregate_numeric_results(run_payloads)
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": None,
        "is_aggregate": True,
        "metric_family": "polygraphscore_classifier",
        "runtime_seconds": float(sum(float(p.get("runtime_seconds", 0.0)) for p in run_payloads)),
        "num_runs": len(run_payloads),
        "run_ids": run_ids,
        "protocol": {
            "base_seed": args.seed,
            "seed_stride": args.seed_stride,
            "reference_split": args.reference_split,
            "max_graphs": args.max_graphs,
            "mode": args.mode,
            "cv_folds": args.cv_folds,
            "num_repeated_partitions": args.num_splits,
        },
        "results": agg["flat"],
        "results_across_runs": agg["nested"],
        "run_results": [
            {"run_id": p.get("run_id"), "seed": (p.get("protocol") or {}).get("seed"), "results": p.get("results", {})}
            for p in run_payloads
        ],
    }
    if output_path is None:
        output_path = aggregate_metric_path(args.dataset, args.model, METRIC_FILENAME)
    save_json(payload, output_path)
    logger.info("Saved across-run PGS aggregate to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-style PolyGraphScore classifier metrics.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--run-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--num-splits", type=int, default=3, help="Repeated PGS fit/test partitions within each trained model run.")
    parser.add_argument("--descriptors", nargs="+", default=None, help="Descriptors: degree clustering spectral orbit4 orbit5 gin attributes concat")
    parser.add_argument("--skip-orbit", action="store_true", help="Backward-compatible alias for --skip-orbits.")
    parser.add_argument("--skip-orbits", action="store_true", help="Skip ORCA orbit descriptors even if listed/defaulted.")
    parser.add_argument("--orca-exec", type=str, default=None)
    parser.add_argument("--classifier", choices=["auto", "tabpfn", "logistic_regression", "logistic", "lr"], default="auto")
    parser.add_argument("--mode", choices=["jsd", "tv"], default="jsd")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--degree-bins", type=int, default=100)
    parser.add_argument("--clustering-bins", type=int, default=100)
    parser.add_argument("--spectral-bins", type=int, default=200)
    parser.add_argument("--max-degree", type=int, default=100)
    parser.add_argument("--gin-dim", type=int, default=128)
    parser.add_argument("--no-attribute-descriptor", action="store_true")
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--node-label-attr", type=str, default="node_label")
    parser.add_argument("--node-feature-attr", type=str, default="feats")
    parser.add_argument("--edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--edge-feature-attr", type=str, default="edge_attr")
    parser.add_argument("--graph-label-attr", type=str, default="graph_label")
    parser.add_argument("--device", type=str, default=None, help="Optional TabPFN device argument.")
    parser.add_argument("--output", type=str, default=None, help="Single-run output path, or aggregate output path for multi-run evaluation.")
    args = parser.parse_args()

    run_ids = parse_run_ids(run_id=args.run_id, run_ids=args.run_ids, num_runs=args.num_runs)
    use_run_paths = should_use_run_paths(run_ids, explicit_run_selection(args.run_id, args.run_ids))
    output_path = Path(args.output) if args.output else None

    payloads = []
    for rid in run_ids:
        seed = run_seed(args.seed, rid, args.seed_stride)
        logical_run_id = rid if use_run_paths else None
        one_output = output_path if len(run_ids) == 1 else None
        payloads.append(_evaluate_one_run(args, run_id=rid, logical_run_id=logical_run_id, seed=seed, output_path=one_output))

    if len(run_ids) > 1 or use_run_paths:
        aggregate_output = output_path if len(run_ids) > 1 else None
        _save_aggregate(args, run_ids, payloads, aggregate_output)


if __name__ == "__main__":
    main()
