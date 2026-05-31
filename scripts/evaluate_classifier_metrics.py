from __future__ import annotations

"""Evaluate PolyGraphScore classifier metrics using the official PolyGraph implementation.

This script keeps the survey benchmark's historical output name
``classifier_metrics.json`` and CLI selectors, but delegates the actual PGS/PGD
computation to ``polygraph-benchmark``.  This avoids the previous local
reimplementation drift: extra attribute descriptors, placeholder GIN features,
non-official orbit aggregation, and classifier probability-column conventions.

Required for paper-style generic PGS:

    export POLYGRAPH_REPO=/path/to/polygraph-benchmark
    export ORCA_EXEC=/path/to/orca

Then, for example:

    PYTHONPATH=src python scripts/evaluate_classifier_metrics.py \
        --dataset planar --model digress --run-ids 0 1 2 --force

The official implementation performs the fit/test split, 4-fold descriptor
selection on the fit split, and held-out test evaluation internally.
"""

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import evaluate_repeated_runs, metric_path
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_FILENAME = "classifier_metrics.json"
DEFAULT_DESCRIPTORS = ["orbit4", "orbit5", "clustering", "degree", "spectral", "gin"]
DEFAULT_MOLECULAR_DESCRIPTORS = ["topochemical", "morgan_fingerprint", "lipinski"]


def _load_official_wrapper_module():
    """Import scripts/evaluate_polygraphscore_official.py robustly.

    When this script is executed directly, the scripts directory is already on
    sys.path.  When imported through empirical_comparison.cli, it may not be.
    """
    try:
        import evaluate_polygraphscore_official as official  # type: ignore

        return official
    except ModuleNotFoundError:
        pass

    path = ROOT / "scripts" / "evaluate_polygraphscore_official.py"
    spec = importlib.util.spec_from_file_location("evaluate_polygraphscore_official", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official PGS wrapper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("evaluate_polygraphscore_official", module)
    spec.loader.exec_module(module)
    return module


OFFICIAL = _load_official_wrapper_module()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _metric_cfg(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    metrics = experiment_cfg.get("metrics", {}) or {}
    block = metrics.get("classifier", {}) or {}
    out = dict(block) if isinstance(block, Mapping) else {}
    out["_source_config_key"] = "metrics.classifier"
    return out


def _merge_cfg(args: argparse.Namespace, experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Merge metrics.classifier with CLI overrides in official-wrapper schema."""
    cfg = _metric_cfg(experiment_cfg)

    # Paper/original-benchmark defaults.  The original benchmark computes
    # StandardPGDInterval with TabPFN v2.5 and the six generic descriptors.
    cfg.setdefault("polygraph_root", None)
    cfg.setdefault("orca_exec", None)
    cfg.setdefault("reference_split", "test")
    cfg.setdefault("metric", "pgs_js_distance")
    cfg.setdefault("variant", "jsd")
    cfg.setdefault("mode", "interval")
    cfg.setdefault("estimate", cfg.get("mode", "interval"))
    cfg.setdefault("classifier", "tabpfn")
    cfg.setdefault("descriptors", DEFAULT_DESCRIPTORS)
    cfg.setdefault("molecular_descriptors", DEFAULT_MOLECULAR_DESCRIPTORS)
    cfg.setdefault("pgs_domain", cfg.get("domain", "auto"))
    cfg.setdefault("molecular_datasets", experiment_cfg.get("molecular_datasets", ["qm9", "zinc"]))
    cfg.setdefault("skip_orbits", False)
    cfg.setdefault("skip_gin", False)
    cfg.setdefault("include_attribute_descriptor", False)
    cfg.setdefault("strip_attributes_for_generic", True)
    cfg.setdefault("num_splits", 1)
    cfg.setdefault("cv_folds", 4)
    cfg.setdefault("num_samples", 10)
    cfg.setdefault("subsample_size", None)
    cfg.setdefault("dataset_root", "outputs/datasets")
    cfg.setdefault("samples_root", "outputs/samples")
    cfg.setdefault("metrics_root", "outputs/metrics")
    cfg.setdefault("allow_orbit_count_fallback", True)
    cfg.setdefault("skip_orbits_if_unavailable", False)
    cfg.setdefault("skip_orbits_when_unavailable", False)
    cfg.setdefault("auto_compile_orca", False)
    cfg.setdefault("gin_device", "cpu")
    cfg.setdefault("gin_seed", 42)
    cfg.setdefault("logistic_max_iter", 5000)

    if experiment_cfg.get("num_reference_graphs") is not None:
        cfg.setdefault("max_reference_graphs", int(experiment_cfg["num_reference_graphs"]))
    if experiment_cfg.get("num_generated_graphs") is not None:
        cfg.setdefault("max_generated_graphs", int(experiment_cfg["num_generated_graphs"]))

    override_keys = [
        "polygraph_root",
        "dataset_root",
        "samples_root",
        "metrics_root",
        "reference_split",
        "classifier",
        "variant",
        "mode",
        "estimate",
        "subsample_size",
        "num_samples",
        "num_splits",
        "max_graphs",
        "max_reference_graphs",
        "max_generated_graphs",
        "degree_bins",
        "clustering_bins",
        "spectral_bins",
        "max_degree",
        "gin_dim",
        "gin_device",
        "logistic_max_iter",
        "orca_exec",
        "cv_folds",
        "pgs_domain",
        "domain",
        "molecular_sanitize",
        "molecular_fingerprint_dim",
        "morgan_dim",
        "rdkit_fingerprint_dim",
        "chemnet_dim",
        "molclr_dim",
        "molclr_batch_size",
        "strip_attributes_for_generic",
        "node_label_attr",
        "node_feature_attr",
        "edge_label_attr",
        "edge_feature_attr",
        "graph_label_attr",
        "attribute_schema_enabled",
    ]
    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value

    if getattr(args, "device", None) is not None and cfg.get("gin_device") is None:
        cfg["gin_device"] = args.device
    if args.descriptors:
        cfg["descriptors"] = list(args.descriptors)
    if args.molecular_descriptors:
        cfg["molecular_descriptors"] = list(args.molecular_descriptors)
    if args.skip_orbits or args.skip_orbit:
        cfg["skip_orbits"] = True
    if args.skip_gin:
        cfg["skip_gin"] = True
    if args.no_attribute_descriptor:
        cfg["include_attribute_descriptor"] = False
        cfg["no_attribute_descriptor"] = True

    # The official generic PGS descriptor set has no benchmark attribute
    # descriptor.  Keep this switch for compatibility, but never add attributes
    # unless a future official implementation supports them explicitly.
    if not bool(cfg.get("include_attribute_descriptor", False)):
        cfg["descriptors"] = [
            d for d in (cfg.get("descriptors") or [])
            if str(d).lower() not in {"attributes", "attribute", "attrs"}
        ]

    cfg["seed"] = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    return cfg


def _score_value(results: Mapping[str, Any], variant: str) -> float:
    candidates = [
        "pgs_js_distance" if variant == "jsd" else "pgs_tv_informedness",
        "pgs",
        "polygraphscore",
        "pgd",
    ]
    for key in candidates:
        value = results.get(key)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    raise RuntimeError(f"Official PGS result has no numeric score in keys {candidates}: {results}")


def _flatten_subscores(subscores: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in dict(subscores or {}).items():
        if isinstance(value, Mapping):
            if isinstance(value.get("mean"), (int, float, np.number)):
                out[str(name)] = float(value["mean"])
        elif isinstance(value, (int, float, np.number)):
            out[str(name)] = float(value)
    return out


def _to_classifier_payload(official_payload: Mapping[str, Any], *, cfg: Mapping[str, Any], started: float) -> dict[str, Any]:
    official_results = dict(official_payload.get("results") or {})
    protocol = dict(official_payload.get("protocol") or {})
    variant = str(protocol.get("variant") or cfg.get("variant", "jsd")).lower()
    score = _score_value(official_results, variant)
    score_key = "pgs_js_distance" if variant == "jsd" else "pgs_tv_informedness"
    descriptor = official_results.get("pgs_descriptor", official_results.get("pgd_descriptor"))
    if descriptor is None and isinstance(official_results.get("pgs_descriptor_frequency"), Mapping):
        freq = official_results.get("pgs_descriptor_frequency") or {}
        if freq:
            descriptor = max(freq, key=lambda k: float(freq[k]))
    subscores = _flatten_subscores(official_results.get("subscores") or {})

    results: dict[str, Any] = {
        score_key: score,
        "polygraphscore": score,
        "pgs": score,
        "pgd": score,
        "pgs_best_descriptor": descriptor,
        "pgs_descriptor": descriptor,
    }

    # Preserve numeric official fields, including split/interval std fields.
    for key, value in official_results.items():
        if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
            results[key] = float(value)
    results[score_key] = score

    for name, value in subscores.items():
        safe = str(name).lower().replace(" ", "_")
        results[f"pgs_subscore_{safe}"] = value
        # Backward-compatible names used by the previous local implementation.
        results[f"pgs_{safe}_test"] = value

    descriptor_summary = {
        name: {
            "test_mean": value,
            "test_std": float(official_results.get(f"pgs_subscore_{name}_std", 0.0) or 0.0),
            "num_partitions": int(protocol.get("num_splits", 1) or 1),
            "source": "official PolyGraphDiscrepancy subscore",
        }
        for name, value in subscores.items()
    }

    converted = {
        "dataset": official_payload.get("dataset"),
        "model": official_payload.get("model"),
        "run_id": official_payload.get("run_id"),
        "metric_family": "polygraphscore_classifier",
        "metric_name": "PGS-JS" if variant == "jsd" else "PGS-TV",
        "implementation_name_in_package": official_payload.get("implementation_name_in_package", "PolyGraphDiscrepancy"),
        "runtime_seconds": time.perf_counter() - started,
        "reference_path": official_payload.get("reference_path"),
        "generated_path": official_payload.get("generated_path"),
        "feature_representation": {
            "name": "official_polygraph_benchmark_descriptors",
            "descriptors_requested": list(cfg.get("descriptors") or DEFAULT_DESCRIPTORS),
            "molecular_descriptors_requested": list(cfg.get("molecular_descriptors") or DEFAULT_MOLECULAR_DESCRIPTORS),
            "attribute_descriptor": "disabled; not part of official generic PGS",
            "generic_descriptor_domain": protocol.get("descriptor_domain", "generic"),
        },
        "classifier": {
            "requested": protocol.get("classifier_requested", cfg.get("classifier")),
            "resolved": protocol.get("classifier_resolved"),
            "object": protocol.get("classifier_object"),
        },
        "protocol": {
            **protocol,
            "implementation": "official polygraph-benchmark via scripts/evaluate_polygraphscore_official.py",
            "compatibility_wrapper": "scripts/evaluate_classifier_metrics.py",
            "metric_filename": METRIC_FILENAME,
            "cv_folds_requested": cfg.get("cv_folds", 4),
            "official_cv_note": "The official implementation controls descriptor CV internally; Algorithm 1 uses 4-fold stratified CV on the fit split.",
        },
        "interpretation": {
            score_key: "Official PolyGraphScore/PGD held-out classifier lower-bound distance in [0, 1]; lower is closer to the reference distribution.",
            "pgs_best_descriptor": "Descriptor selected by the official implementation using the fit/CV score; final score is measured on held-out test graphs.",
        },
        "results": results,
        "descriptor_summary": descriptor_summary,
        "official_results": official_results,
        "result_diagnostics": official_payload.get("result_diagnostics", {}),
        "dependency_shims": official_payload.get("dependency_shims", {}),
        "polygraph_root_resolved": official_payload.get("polygraph_root_resolved"),
        "versions": official_payload.get("versions", {}),
    }
    return converted


def _evaluate(args: argparse.Namespace, *, seed: int, output_path: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    root = ROOT
    cfg_path = Path(args.experiment_config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    experiment_cfg = _load_yaml(cfg_path)
    cfg = _merge_cfg(args, experiment_cfg)
    cfg["seed"] = int(seed)

    descriptors = OFFICIAL.canonical_descriptors(cfg)
    # Molecular datasets use molecule descriptors, so do not require orbit-count
    # just to import the generic descriptor module.
    if OFFICIAL.pgs_domain_for_dataset(args.dataset, cfg) == "molecular":
        descriptors = [d for d in descriptors if not d.startswith("orbit")]

    descriptors = OFFICIAL.prepare_imports(
        cfg,
        descriptors,
        str(cfg.get("classifier", "tabpfn")),
        survey_root=root,
    )
    poly = OFFICIAL.import_polygraph_objects(
        str(cfg.get("polygraph_root")) if cfg.get("polygraph_root") else None,
        survey_root=root,
    )
    try:
        classifier, classifier_resolved = OFFICIAL.make_classifier(str(cfg.get("classifier", "tabpfn")), cfg, seed, poly)
    except TypeError:
        # Backward compatibility with older helper versions.
        classifier, classifier_resolved = OFFICIAL.make_classifier(str(cfg.get("classifier", "tabpfn")), cfg, seed)

    official_payload = OFFICIAL.evaluate_one(
        args.dataset,
        args.model,
        args.run_id,
        cfg,
        root,
        poly,
        descriptors,
        classifier,
        classifier_resolved,
        seed,
        output_path=None,
    )
    payload = _to_classifier_payload(official_payload, cfg=cfg, started=started)
    if output_path is None:
        output_path = metric_path(args.dataset, args.model, METRIC_FILENAME, run_id=args.run_id)
    save_json(payload, output_path)
    logger.info(
        "Saved official PGS classifier metrics: dataset=%s model=%s run_id=%s pgs=%.6f -> %s",
        args.dataset,
        args.model,
        args.run_id,
        float(payload["results"].get("pgs_js_distance", payload["results"].get("pgs", float("nan")))),
        output_path,
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official PolyGraphScore classifier metric.")
    parser.add_argument("--experiment-config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--samples-root", type=str, default=None)
    parser.add_argument("--metrics-root", type=str, default=None)
    parser.add_argument("--polygraph-root", type=str, default=None, help="Optional override; normally use POLYGRAPH_REPO.")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-graphs", type=int, default=None, help="Cap applied to both reference and generated graphs unless specific caps are set.")
    parser.add_argument("--max-reference-graphs", type=int, default=None)
    parser.add_argument("--max-generated-graphs", type=int, default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--run-ids", type=int, nargs="+", default=None)
    parser.add_argument("--mode", choices=["point", "interval", "subsampling"], default=None)
    parser.add_argument("--estimate", dest="mode", choices=["point", "interval", "subsampling"], default=None)
    parser.add_argument("--variant", choices=["jsd", "informedness"], default=None)
    parser.add_argument("--num-splits", type=int, default=None, help="Repeated point estimates; ignored in interval mode.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of interval subsamples in official PolyGraphDiscrepancyInterval.")
    parser.add_argument("--subsample-size", type=int, default=None, help="Per-class interval sample size. Default follows original benchmark: min(n/2, 2048).")
    parser.add_argument("--descriptors", nargs="+", default=None, help="Official generic descriptors: degree clustering spectral orbit4 orbit5 gin")
    parser.add_argument("--molecular-descriptors", nargs="+", default=None, help="Official molecule descriptors, e.g. topochemical morgan_fingerprint lipinski chemnet molclr")
    parser.add_argument("--pgs-domain", choices=["auto", "generic", "molecular"], default=None)
    parser.add_argument("--domain", dest="pgs_domain", choices=["auto", "generic", "molecular"], default=None)
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--skip-orbits", action="store_true")
    parser.add_argument("--skip-gin", action="store_true")
    parser.add_argument("--orca-exec", type=str, default=None, help="Optional override; normally use ORCA_EXEC.")
    parser.add_argument("--classifier", choices=["official_default", "default", "auto", "tabpfn", "tabpfn-v25", "logistic_regression", "logistic", "lr"], default=None)
    parser.add_argument("--cv-folds", type=int, default=None, help="Accepted for compatibility; official PGS uses 4-fold CV internally.")
    parser.add_argument("--degree-bins", type=int, default=None, help="Accepted for compatibility; official SparseDegreeHistogram controls degree support.")
    parser.add_argument("--clustering-bins", type=int, default=None)
    parser.add_argument("--spectral-bins", type=int, default=None)
    parser.add_argument("--max-degree", type=int, default=None, help="Accepted for compatibility; official SparseDegreeHistogram controls degree support.")
    parser.add_argument("--gin-dim", type=int, default=None, help="Accepted for compatibility; official RandomGIN controls embedding width.")
    parser.add_argument("--gin-device", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Compatibility alias for --gin-device when unset.")
    parser.add_argument("--logistic-max-iter", type=int, default=None)
    parser.add_argument("--no-attribute-descriptor", action="store_true", help="Kept for compatibility; attributes are disabled by default because they are not part of official generic PGS.")
    parser.add_argument("--attribute-schema-enabled", choices=["auto", "true", "false"], default=None, help="Accepted for compatibility; official generic PGS ignores benchmark attributes.")
    parser.add_argument("--node-label-attr", type=str, default=None)
    parser.add_argument("--node-feature-attr", type=str, default=None)
    parser.add_argument("--edge-label-attr", type=str, default=None)
    parser.add_argument("--edge-feature-attr", type=str, default=None)
    parser.add_argument("--graph-label-attr", type=str, default=None)
    parser.add_argument("--molecular-sanitize", choices=["true", "false"], default=None)
    parser.add_argument("--molecular-fingerprint-dim", type=int, default=None)
    parser.add_argument("--morgan-dim", type=int, default=None)
    parser.add_argument("--rdkit-fingerprint-dim", type=int, default=None)
    parser.add_argument("--chemnet-dim", type=int, default=None)
    parser.add_argument("--molclr-dim", type=int, default=None)
    parser.add_argument("--molclr-batch-size", type=int, default=None)
    parser.add_argument("--strip-attributes-for-generic", type=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Accepted for run_benchmark compatibility; this script always overwrites its metric file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg_path = Path(args.experiment_config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    experiment_cfg = _load_yaml(cfg_path)
    base_seed = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    output_path = Path(args.output) if args.output else None
    evaluate_repeated_runs(args, metric_filename=METRIC_FILENAME, evaluate_fn=_evaluate, base_seed=base_seed, output_path=output_path)


if __name__ == "__main__":
    main()
