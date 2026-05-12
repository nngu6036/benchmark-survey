from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits
from empirical_comparison.evaluation.run_utils import (
    make_model_config,
    run_output_dir,
)
from empirical_comparison.graphs.attributes import attribute_coverage, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.registry import get_model_class, available_datasets, available_models
from empirical_comparison.utils.compute import PeakMemoryMonitor, compute_report
from empirical_comparison.utils.io import load_yaml, save_json, save_yaml, stable_hash
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.seed import set_seed

logger = get_logger(__name__)


def _train_one_run(
    *,
    model_name: str,
    dataset: str,
    base_model_cfg: dict,
    model_cfg_path: Path,
    data_cfg_path: Path,
    dataset_root: str,
    seed: int,
    force_data: bool,
    dry_run: bool,
) -> dict:
    set_seed(seed)
    model_cfg = make_model_config(
        base_model_cfg,
        dataset=dataset,
        model=model_name,
        seed=seed,
    )

    logger.info(
        "Training model=%s dataset=%s seed=%s checkpoint=%s",
        model_name,
        dataset,
        seed,
        model_cfg.get("checkpoint_path"),
    )
    if dry_run:
        return {
            "dataset": dataset,
            "model": model_name,
            "seed": seed,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "dry_run": True,
        }

    splits = load_dataset_splits(
        dataset,
        output_root=dataset_root,
        build_if_missing=True,
        config_path=data_cfg_path,
        force=force_data,
    )
    attr_schema = normalize_schema(model_cfg)
    all_graphs_for_attrs = list(splits.get("train", [])) + list(splits.get("val", [])) + list(splits.get("test", []))
    all_attr_stats = fit_attribute_statistics(all_graphs_for_attrs, attr_schema)
    splits = {
        split_name: canonicalize_graph_attributes(list(graphs), attr_schema, all_attr_stats)[0]
        for split_name, graphs in splits.items()
    }
    train_attr_stats = fit_attribute_statistics(list(splits.get("train", [])), attr_schema)
    model_cfg["graph_attribute_metadata"] = {
        "schema": attr_schema,
        "all_attribute_stats": all_attr_stats.to_dict(),
        "train_attribute_stats": train_attr_stats.to_dict(),
        "coverage": {k: attribute_coverage(list(v), attr_schema) for k, v in splits.items()},
        "native_attribute_note": "Wrappers may support only a subset of these attributes; generated metadata records fallback postprocessing when used.",
    }
    model_cfg["graph_attribute_stats"] = train_attr_stats.to_dict()
    model_cls = get_model_class(model_name)
    model = model_cls(model_cfg)
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        model.train(splits["train"], splits.get("val"), splits.get("test"))
    elapsed = time.perf_counter() - start
    compute = compute_report(
        operation="training",
        runtime_seconds=elapsed,
        num_graphs=len(splits.get("train", [])),
        memory=memory_monitor.to_dict(),
    )

    run_dir = run_output_dir(dataset, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(model_cfg, run_dir / "resolved_model_config.yaml", force=True)
    metadata = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "runtime_seconds": elapsed,
        "training_time_seconds": elapsed,
        "training_time_minutes": elapsed / 60.0,
        "hardware": compute["hardware_label"],
        "peak_memory_mib": compute.get("peak_memory_mib"),
        "compute": compute,
        "compute_budget": {
            "dataset": dataset,
            "model": model_name,
            "hardware": compute["hardware_label"],
            "training_time_seconds": elapsed,
            "training_time_minutes": elapsed / 60.0,
            "sampling_time_per_128_graphs_seconds": None,
            "peak_memory_mib": compute.get("peak_memory_mib"),
            "notes": "Training run measured by scripts/train_model.py.",
        },
        "checkpoint_path": model_cfg.get("checkpoint_path"),
        "model_config_path": str(model_cfg_path),
        "model_config_hash": stable_hash(model_cfg),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "graph_attributes": model_cfg.get("graph_attribute_metadata", {}),
        "capabilities": model_cls.capabilities() if hasattr(model_cls, "capabilities") else {},
    }
    save_json(metadata, run_dir / "train_metadata.json", force=True)
    logger.info(
        "Finished training model=%s dataset=%s in %.2fs",
        model_name,
        dataset,
        elapsed,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph generator wrapper(s) on persisted benchmark splits.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-data", action="store_true", help="Rebuild persisted dataset splits before training.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_cfg_path = Path(args.model_config) if args.model_config else Path("configs/models") / f"{args.model}.yaml"
    data_cfg_path = Path(args.dataset_config) if args.dataset_config else Path("configs/datasets") / f"{args.dataset}.yaml"
    base_model_cfg = load_yaml(model_cfg_path)

    logger.info("model=%s dataset=%s seed=%s", args.model, args.dataset, args.seed)
    logger.info("model_config=%s dataset_config=%s", model_cfg_path, data_cfg_path)

    _train_one_run(
        model_name=args.model,
        dataset=args.dataset,
        base_model_cfg=base_model_cfg,
        model_cfg_path=model_cfg_path,
        data_cfg_path=data_cfg_path,
        dataset_root=args.dataset_root,
        seed=args.seed,
        force_data=args.force_data,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
