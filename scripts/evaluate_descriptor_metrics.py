from __future__ import annotations
import argparse, pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.metrics.descriptor.descriptors import degree_histogram, clustering_histogram, orbit_placeholder, spectral_histogram
from empirical_comparison.metrics.descriptor.mmd import mmd_unbiased
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
    res = {
        "degree_mmd": mmd_unbiased([degree_histogram(g) for g in ref_graphs], [degree_histogram(g) for g in gen_graphs]),
        "clustering_mmd": mmd_unbiased([clustering_histogram(g) for g in ref_graphs], [clustering_histogram(g) for g in gen_graphs]),
        "orbit_mmd": mmd_unbiased([orbit_placeholder(g) for g in ref_graphs], [orbit_placeholder(g) for g in gen_graphs]),
        "spectral_mmd": mmd_unbiased([spectral_histogram(g) for g in ref_graphs], [spectral_histogram(g) for g in gen_graphs]),
    }
    save_json({"dataset": args.dataset, "model": args.model, "results": res}, Path("outputs/metrics") / args.dataset / args.model / "descriptor_metrics.json")
    logger.info("Saved descriptor metrics for %s on %s", args.model, args.dataset)

if __name__ == "__main__":
    main()
