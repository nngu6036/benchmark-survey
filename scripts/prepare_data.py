from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import build_dataset_splits, save_dataset_splits
from empirical_comparison.registry import available_datasets
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist benchmark dataset splits.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing persisted splits and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned outputs without writing files.")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path("configs/datasets") / f"{args.dataset}.yaml"
    cfg = load_yaml(cfg_path)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = Path(args.output_root) / args.dataset
    logger.info("dataset=%s config=%s seed=%s output=%s", args.dataset, cfg_path, seed, out_dir)
    if args.dry_run:
        logger.info("Dry run: would build %s and write train/val/test splits to %s", args.dataset, out_dir)
        return

    start = time.perf_counter()
    splits = build_dataset_splits(args.dataset, cfg)
    save_dataset_splits(args.dataset, splits, cfg, output_root=args.output_root, force=args.force)
    elapsed = time.perf_counter() - start
    logger.info("Prepared dataset %s with sizes: %s", args.dataset, {k: len(v) for k, v in splits.items()})
    logger.info("Saved dataset artifacts to %s in %.2fs", out_dir, elapsed)


if __name__ == "__main__":
    main()
