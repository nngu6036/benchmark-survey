from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def _load_generated_graphs(dataset: str, model: str) -> list:
    sample_file = existing_sample_path(dataset, model)
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


def _pgs_js_mean_std(split_payloads: list[dict]) -> dict[str, float]:
    values: list[float] = []
    for payload in split_payloads:
        results = payload.get("results") or {}
        value = results.get("pgs_js_distance", results.get("polygraphscore"))
        if isinstance(value, (int, float, np.number)):
            values.append(float(value))
    if not values:
        raise RuntimeError("PGS-JS evaluation did not produce any numeric pgs_js_distance values.")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "pgs_js_distance": mean,
        "pgs_js_distance_split_mean": mean,
        "pgs_js_distance_split_std": float(arr.std(ddof=0)),
    }


def _strip_to_pgs_js(payload: dict) -> dict:
    results = payload.get("results") or {}
    score = results.get("pgs_js_distance", results.get("polygraphscore"))
    clean_descriptors = []
    for item in payload.get("descriptor_results", []):
        clean_descriptors.append({
            "descriptor": item.get("descriptor"),
            "cv_score": item.get("cv_score"),
            "cv_score_std": item.get("cv_score_std"),
            "test_score": item.get("test_score"),
            "classifier": item.get("classifier"),
            "num_fit_graphs_per_class": item.get("num_fit_graphs_per_class"),
            "num_test_graphs_per_class": item.get("num_test_graphs_per_class"),
            "feature_dim": item.get("feature_dim"),
        })
    return {
        "results": {"pgs_js_distance": float(score)} if isinstance(score, (int, float, np.number)) else {},
        "descriptor_results": clean_descriptors,
        "best_descriptor": payload.get("best_descriptor"),
        "classifier": payload.get("classifier"),
        "skipped_descriptors": payload.get("skipped_descriptors", {}),
    }


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


def _evaluate(args, *, seed: int, output_path: Path | None) -> dict:
    start = time.perf_counter()
    ref_graphs = _subsample(_load_reference_graphs(args.dataset, args.dataset_root, args.reference_split), args.max_graphs, seed)
    gen_graphs = _subsample(_load_generated_graphs(args.dataset, args.model), args.max_graphs, seed + 1)

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
        "Evaluating PGS-JS: dataset=%s model=%s ref_split=%s ref=%d gen=%d partitions=%d classifier=%s",
        args.dataset,
        args.model,
        args.reference_split,
        len(ref_graphs),
        len(gen_graphs),
        args.num_splits,
        args.classifier,
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
            classifier=args.classifier,
            cv_folds=args.cv_folds,
            seed=split_seed,
            skip_unavailable=True,
            device=args.device,
        )
        pgs_payload = _strip_to_pgs_js(pgs_payload)
        pgs_payload["split_id"] = split_id
        pgs_payload["seed"] = split_seed
        split_payloads.append(pgs_payload)
        logger.info(
            "pgs_js_partition=%d pgs_js=%.4f selected=%s classifier=%s",
            split_id,
            float(pgs_payload["results"].get("pgs_js_distance", float("nan"))),
            pgs_payload.get("best_descriptor"),
            pgs_payload.get("classifier"),
        )

    results = _pgs_js_mean_std(split_payloads)

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "polygraphscore_classifier",
        "runtime_seconds": time.perf_counter() - start,
        "metric_name": "PGS-JS",
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
            "mode": "jsd",
            "seed": seed,
        },
        "graph_attributes": {
            "schema": attr_schema,
            "reference_attribute_coverage": attribute_coverage(ref_graphs, attr_schema),
            "generated_attribute_coverage": attribute_coverage(gen_graphs, attr_schema),
        },
        "interpretation": {
            "pgs_js_distance": "Paper-style PGS-JS distance lower-bound estimate in [0, 1]; lower is closer to the reference distribution.",
            "pgs_best_descriptor": "Descriptor selected by highest cross-validation score on the fit set for each partition.",
        },
        "results": results,
        "descriptor_summary": _descriptor_summary(split_payloads),
        "split_results": split_payloads,
    }
    if output_path is None:
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME)
    save_json(payload, output_path)
    logger.info("Saved PGS metrics to %s", output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-style PGS-JS classifier metric.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--num-splits", type=int, default=3, help="Repeated PGS fit/test partitions for the sampled graph set.")
    parser.add_argument("--descriptors", nargs="+", default=None, help="Descriptors: degree clustering spectral orbit4 orbit5 gin attributes concat")
    parser.add_argument("--skip-orbit", action="store_true", help="Backward-compatible alias for --skip-orbits.")
    parser.add_argument("--skip-orbits", action="store_true", help="Skip ORCA orbit descriptors even if listed/defaulted.")
    parser.add_argument("--orca-exec", type=str, default=None)
    parser.add_argument("--classifier", choices=["auto", "tabpfn", "logistic_regression", "logistic", "lr"], default="auto")
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    _evaluate(args, seed=args.seed, output_path=output_path)


if __name__ == "__main__":
    main()
