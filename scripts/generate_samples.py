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
    sample_config_path,
    sample_metadata_path,
    sample_path,
)
from empirical_comparison.generation.sampler import model_capabilities, sample_graphs
from empirical_comparison.graphs.attributes import apply_empirical_attributes, attribute_coverage, fit_attribute_statistics, normalize_schema
from empirical_comparison.generation.validity import quality_metrics
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.io import load_yaml, save_json, save_pickle, save_yaml, stable_hash
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.seed import set_seed

logger = get_logger(__name__)


def _generate_samples(
    *,
    model_name: str,
    dataset: str,
    base_cfg: dict,
    cfg_path: Path,
    dataset_root: str,
    num_samples: int,
    seed: int,
    sample_seed_offset: int,
    force: bool,
    dry_run: bool,
    show_progress: bool,
) -> dict:
    seed = int(seed) + int(sample_seed_offset)
    set_seed(seed)
    cfg = make_model_config(
        base_cfg,
        dataset=dataset,
        model=model_name,
        seed=seed,
    )
    cfg["num_samples"] = int(num_samples)

    out = sample_path(dataset, model_name)
    metadata_out = sample_metadata_path(dataset, model_name)
    resolved_cfg_out = sample_config_path(dataset, model_name)
    logger.info(
        "Generating samples: dataset=%s model=%s num_samples=%d seed=%d checkpoint=%s",
        dataset,
        model_name,
        num_samples,
        seed,
        cfg.get("checkpoint_path"),
    )
    if dry_run:
        logger.info("Dry run: would write %s", out)
        return {"dataset": dataset, "model": model_name, "seed": seed, "sample_path": str(out), "dry_run": True}
    if out.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {out}. Use --force to overwrite.")

    start = time.perf_counter()
    graphs = sample_graphs(
        model_name,
        cfg,
        num_samples,
        seed=seed,
        show_progress=show_progress,
        progress_desc=f"Sampling {dataset}/{model_name}",
    )
    elapsed = time.perf_counter() - start
    attr_schema = normalize_schema(cfg)
    attr_postprocess_applied = False
    try:
        splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
        ref_graphs = list(splits.get("train", []))
    except Exception:
        ref_graphs = []
    attr_stats = fit_attribute_statistics(ref_graphs, attr_schema) if ref_graphs else fit_attribute_statistics([], attr_schema)
    attr_strategy = str(attr_schema.get("generated_attribute_strategy", "empirical")).lower()
    if attr_stats.has_any_attributes and attr_strategy == "empirical":
        before_cov = attribute_coverage(graphs, attr_schema)
        overwrite = bool(attr_schema.get("overwrite_generated_attributes", False))
        if overwrite or not before_cov.get("has_any_attributes", False):
            graphs = apply_empirical_attributes(graphs, attr_stats, seed=seed, overwrite=overwrite)
            attr_postprocess_applied = True
    quality = quality_metrics(graphs, reference_graphs=ref_graphs, dataset=dataset)

    save_pickle(graphs, out, force=force)
    metadata = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "num_samples_requested": num_samples,
        "num_samples_saved": len(graphs),
        "runtime_seconds": elapsed,
        "seconds_per_graph": elapsed / max(len(graphs), 1),
        "sample_path": str(out),
        "checkpoint_path": cfg.get("checkpoint_path"),
        "model_config_path": str(cfg_path),
        "model_config_hash": stable_hash(cfg),
        "capabilities": model_capabilities(model_name),
        "quality": quality,
        "graph_attributes": {
            "schema": attr_schema,
            "fallback_attribute_postprocessing_applied": attr_postprocess_applied,
            "fallback_note": "If true, attributes were attached from empirical training-set marginals by the benchmark, not generated natively by the upstream model.",
            "train_attribute_stats": attr_stats.to_dict(),
            "train_attribute_coverage": attribute_coverage(ref_graphs, attr_schema) if ref_graphs else {},
            "generated_attribute_coverage": attribute_coverage(graphs, attr_schema),
        },
    }
    save_json(metadata, metadata_out, force=True)
    save_yaml(cfg, resolved_cfg_out, force=True)
    logger.info("Saved %d generated graphs to %s in %.2fs", len(graphs), out, elapsed)
    logger.info("Saved sample metadata to %s", metadata_out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate samples from a trained graph generator wrapper.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed-offset", type=int, default=17, help="Offset added to the training seed for sampling.")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the sampling progress bar.")
    args = parser.parse_args()

    cfg_path = Path(args.model_config) if args.model_config else Path("configs/models") / f"{args.model}.yaml"
    base_cfg = load_yaml(cfg_path)
    _generate_samples(
        model_name=args.model,
        dataset=args.dataset,
        base_cfg=base_cfg,
        cfg_path=cfg_path,
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_seed_offset=args.sample_seed_offset,
        force=args.force,
        dry_run=args.dry_run,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
