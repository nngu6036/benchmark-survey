from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import build_dataset_splits, save_dataset_splits
from empirical_comparison.registry import available_datasets
from empirical_comparison.utils.io import load_pickle, load_yaml
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.seed import set_seed

logger = get_logger(__name__)


def _node_label_value(data: dict[str, Any]) -> Any:
    return data.get("node_label", data.get("atom_type", data.get("label")))


def _node_feature_scalar(data: dict[str, Any]) -> Any:
    value = data.get("feats", data.get("feature", data.get("x")))
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        if hasattr(value, "reshape"):
            arr = value.reshape(-1)
            if len(arr):
                item = arr[0]
                return item.item() if hasattr(item, "item") else item
    except Exception:
        return None
    return value


def _summarize_atom_types(splits: dict[str, list], *, title: str) -> None:
    counts: dict[str, Counter] = {}
    feature_examples: dict[Any, set[str]] = defaultdict(set)
    for split, graphs in splits.items():
        counter = Counter()
        for graph in graphs:
            for _, data in graph.nodes(data=True):
                label = _node_label_value(data)
                counter[label] += 1
                feature = _node_feature_scalar(data)
                if feature is not None and len(feature_examples[label]) < 5:
                    feature_examples[label].add(str(feature))
        counts[split] = counter
    logger.info("%s atom_type/node_label counts:", title)
    for split, counter in counts.items():
        logger.info("  %s: %s", split, dict(sorted(counter.items(), key=lambda item: str(item[0]))))
    if feature_examples:
        mapping = {label: sorted(values) for label, values in sorted(feature_examples.items(), key=lambda item: str(item[0]))}
        logger.info("  node_label -> example first feature values: %s", mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist benchmark dataset splits.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/datasets")
    parser.add_argument("--download-root", type=str, default=None, help="Override raw download/cache root for real PyG datasets such as QM9 and ZINC.")
    parser.add_argument("--max-graphs", type=int, default=None, help="Optional cap on the number of raw graphs converted before splitting.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing persisted splits and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned outputs without writing files.")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path("configs/datasets") / f"{args.dataset}.yaml"
    cfg = load_yaml(cfg_path)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.download_root is not None:
        cfg["pyg_root"] = args.download_root
    if args.max_graphs is not None:
        cfg["max_graphs"] = int(args.max_graphs)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = Path(args.output_root) / args.dataset
    logger.info("dataset=%s config=%s seed=%s output=%s", args.dataset, cfg_path, seed, out_dir)
    if args.dry_run:
        logger.info("Dry run: would build %s and write train/val/test splits to %s", args.dataset, out_dir)
        return

    start = time.perf_counter()
    splits = build_dataset_splits(args.dataset, cfg)
    if args.dataset.lower() == "zinc":
        _summarize_atom_types(splits, title="Raw ZINC")
    save_dataset_splits(args.dataset, splits, cfg, output_root=args.output_root, force=args.force)
    if args.dataset.lower() == "zinc":
        persisted = {
            split: load_pickle(Path(args.output_root) / args.dataset / f"{split}.pkl")
            for split in ("train", "val", "test")
        }
        _summarize_atom_types(persisted, title="Canonical ZINC")
    elapsed = time.perf_counter() - start
    logger.info("Prepared dataset %s with sizes: %s", args.dataset, {k: len(v) for k, v in splits.items()})
    logger.info("Saved dataset artifacts to %s in %.2fs", out_dir, elapsed)


if __name__ == "__main__":
    main()
