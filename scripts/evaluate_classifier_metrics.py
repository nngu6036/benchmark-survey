from __future__ import annotations
import argparse, pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.metrics.classifier.dataset import make_binary_dataset
from empirical_comparison.metrics.classifier.score import separation_score
from empirical_comparison.metrics.classifier.train import LogisticRegressionPlaceholder
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
    enc = RandomGINPlaceholder(feature_dim=64, seed=123)
    ref_feats = [enc.encode(g) for g in ref_graphs]
    gen_feats = [enc.encode(g) for g in gen_graphs]
    x, y = make_binary_dataset(ref_feats, gen_feats)
    n = len(y) // 2
    clf = LogisticRegressionPlaceholder().fit(x[:n], y[:n])
    probs = clf.predict_score(x[n:])
    score = separation_score(probs, y[n:])
    save_json({"dataset": args.dataset, "model": args.model, "results": {"classifier_separation": float(score)}}, Path("outputs/metrics") / args.dataset / args.model / "classifier_metrics.json")
    logger.info("Saved classifier-based metrics for %s on %s", args.model, args.dataset)

if __name__ == "__main__":
    main()
