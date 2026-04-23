from __future__ import annotations
import argparse, pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.metrics.learned_feature.distance import feature_mmd
from empirical_comparison.metrics.learned_feature.encoder import RandomGINPlaceholder
from empirical_comparison.registry import DATASET_REGISTRY
from empirical_comparison.utils.io import load_yaml, save_json
from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    data_cfg = load_yaml(Path("configs/datasets") / f"{args.dataset}.yaml")
    ref_graphs = DATASET_REGISTRY[args.dataset](data_cfg).build()["test"]
    with open(Path("outputs/samples") / args.dataset / f"{args.model}.pkl", "rb") as f: gen_graphs = pickle.load(f)
    enc = RandomGINPlaceholder(feature_dim=128, seed=42)
    ref_feats = [enc.encode(g) for g in ref_graphs]
    gen_feats = [enc.encode(g) for g in gen_graphs]
    score = feature_mmd(ref_feats, gen_feats)
    save_json({"dataset": args.dataset, "model": args.model, "results": {"learned_feature_mmd": score}}, Path("outputs/metrics") / args.dataset / args.model / "learned_feature_metrics.json")
    logger.info("Saved learned-feature metrics for %s on %s", args.model, args.dataset)

if __name__ == "__main__":
    main()
