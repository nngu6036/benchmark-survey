from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import DATASET_REGISTRY
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["sbm", "planar"])
    args = parser.parse_args()
    cfg = load_yaml(Path("configs/datasets") / f"{args.dataset}.yaml")
    builder = DATASET_REGISTRY[args.dataset](cfg)
    splits = builder.build()
    logger.info("Prepared dataset %s with sizes: %s", args.dataset, {k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()
