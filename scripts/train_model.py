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
    explicit_run_selection,
    make_model_run_config,
    parse_run_ids,
    run_output_dir,
    run_seed,
    should_use_run_paths,
)
from empirical_comparison.graphs.attributes import attribute_coverage, canonicalize_graph_attributes, fit_attribute_statistics, normalize_schema
from empirical_comparison.registry import get_model_class, available_datasets, available_models
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
    base_seed: int,
    seed_stride: int,
    run_id: int,
    use_run_paths: bool,
    force_data: bool,
    dry_run: bool,
) -> dict:
    seed = run_seed(base_seed, run_id, seed_stride)
    logical_run_id = run_id if use_run_paths else None
    set_seed(seed)
    model_cfg = make_model_run_config(
        base_model_cfg,
        dataset=dataset,
        model=model_name,
        run_id=logical_run_id,
        seed=seed,
        use_run_paths=use_run_paths,
    )

    logger.info(
        "Training model=%s dataset=%s run_id=%s seed=%s checkpoint=%s",
        model_name,
        dataset,
        "legacy" if logical_run_id is None else logical_run_id,
        seed,
        model_cfg.get("checkpoint_path"),
    )
    if dry_run:
        return {
            "dataset": dataset,
            "model": model_name,
            "run_id": logical_run_id,
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
    model.train(splits["train"], splits.get("val"), splits.get("test"))
    elapsed = time.perf_counter() - start

    run_dir = run_output_dir(dataset, model_name, logical_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(model_cfg, run_dir / "resolved_model_config.yaml", force=True)
    metadata = {
        "dataset": dataset,
        "model": model_name,
        "run_id": logical_run_id,
        "seed": seed,
        "runtime_seconds": elapsed,
        "checkpoint_path": model_cfg.get("checkpoint_path"),
        "model_config_path": str(model_cfg_path),
        "model_config_hash": stable_hash(model_cfg),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "graph_attributes": model_cfg.get("graph_attribute_metadata", {}),
        "capabilities": model_cls.capabilities() if hasattr(model_cls, "capabilities") else {},
    }
    save_json(metadata, run_dir / "train_metadata.json", force=True)
    logger.info(
        "Finished training model=%s dataset=%s run_id=%s in %.2fs",
        model_name,
        dataset,
        "legacy" if logical_run_id is None else logical_run_id,
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
    parser.add_argument("--seed", type=int, default=42, help="Base seed. Run i uses seed + i * seed_stride.")
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--num-runs", type=int, default=1, help="Train this many independent model versions.")
    parser.add_argument("--run-id", type=int, default=None, help="Train only one explicit run id.")
    parser.add_argument("--run-ids", nargs="+", type=int, default=None, help="Train specific run ids.")
    parser.add_argument("--force-data", action="store_true", help="Rebuild persisted dataset splits before training.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_cfg_path = Path(args.model_config) if args.model_config else Path("configs/models") / f"{args.model}.yaml"
    data_cfg_path = Path(args.dataset_config) if args.dataset_config else Path("configs/datasets") / f"{args.dataset}.yaml"
    base_model_cfg = load_yaml(model_cfg_path)
    run_ids = parse_run_ids(run_id=args.run_id, run_ids=args.run_ids, num_runs=args.num_runs)
    use_run_paths = should_use_run_paths(run_ids, explicit_run_selection(args.run_id, args.run_ids))

    logger.info("model=%s dataset=%s runs=%s base_seed=%s", args.model, args.dataset, run_ids, args.seed)
    logger.info("model_config=%s dataset_config=%s run_aware_paths=%s", model_cfg_path, data_cfg_path, use_run_paths)

    metadata = []
    for rid in run_ids:
        metadata.append(
            _train_one_run(
                model_name=args.model,
                dataset=args.dataset,
                base_model_cfg=base_model_cfg,
                model_cfg_path=model_cfg_path,
                data_cfg_path=data_cfg_path,
                dataset_root=args.dataset_root,
                base_seed=args.seed,
                seed_stride=args.seed_stride,
                run_id=rid,
                use_run_paths=use_run_paths,
                force_data=args.force_data,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run and use_run_paths:
        out = Path("outputs/runs") / args.dataset / args.model / "training_runs.json"
        save_json(
            {
                "dataset": args.dataset,
                "model": args.model,
                "base_seed": args.seed,
                "seed_stride": args.seed_stride,
                "num_runs": len(run_ids),
                "run_ids": run_ids,
                "runs": metadata,
            },
            out,
            force=True,
        )
        logger.info("Saved repeated-training manifest to %s", out)


if __name__ == "__main__":
    main()
