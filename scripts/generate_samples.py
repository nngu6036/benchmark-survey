from __future__ import annotations
import argparse, pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.generation.sampler import sample_graphs
from empirical_comparison.utils.io import load_yaml
from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_yaml(Path("configs/models") / f"{args.model}.yaml")
    graphs = sample_graphs(args.model, cfg, args.num_samples, seed=args.seed)
    out = Path("outputs/samples") / args.dataset / f"{args.model}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f: pickle.dump(graphs, f)
    logger.info("Saved %d generated graphs to %s", len(graphs), out)

if __name__ == "__main__":
    main()
