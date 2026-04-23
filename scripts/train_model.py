from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import DATASET_REGISTRY, MODEL_REGISTRY
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    args = parser.parse_args()
    model_cfg = load_yaml(Path("configs/models") / f"{args.model}.yaml")
    data_cfg = load_yaml(Path("configs/datasets") / f"{args.dataset}.yaml")
    splits = DATASET_REGISTRY[args.dataset](data_cfg).build()
    model = MODEL_REGISTRY[args.model](model_cfg)
    model.train(splits["train"], splits["val"])
    logger.info("Finished placeholder training for %s on %s", args.model, args.dataset)

if __name__ == "__main__":
    main()
