from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

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


def _load_generated_graphs(dataset: str, model: str) -> list:
    sample_path = Path("outputs/samples") / dataset / f"{model}.pkl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_path}. Run generate_samples.py first.")
    with open(sample_path, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs to be stored as a list, got {type(graphs)}.")
    if len(graphs) == 0:
        raise ValueError(f"No generated graphs found in {sample_path}.")
    return graphs


def _load_reference_graphs(dataset: str) -> list:
    data_cfg_path = Path("configs/datasets") / f"{dataset}.yaml"
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_cfg_path}")
    data_cfg = load_yaml(data_cfg_path)
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset}' is not registered. Available datasets: {sorted(DATASET_REGISTRY.keys())}")
    splits = DATASET_REGISTRY[dataset](data_cfg).build()
    if "test" not in splits:
        raise KeyError(f"Dataset '{dataset}' did not return a 'test' split. Available splits: {sorted(splits.keys())}")
    graphs = list(splits["test"])
    if len(graphs) == 0:
        raise ValueError(f"Test split for dataset '{dataset}' is empty.")
    return graphs


def _maybe_subsample(graphs: Sequence, max_graphs: int | None, seed: int) -> list:
    graphs = list(graphs)
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def _encode_graphs(graphs: Sequence, encoder: RandomGINPlaceholder) -> np.ndarray:
    feats = [encoder.encode(g) for g in graphs]
    feats_arr = np.asarray(feats, dtype=np.float64)
    if feats_arr.ndim != 2:
        raise ValueError(f"Encoded features must be 2D, got shape {feats_arr.shape}")
    if not np.all(np.isfinite(feats_arr)):
        raise ValueError("Encoded features contain NaN or Inf values.")
    return feats_arr


def _feature_summary(features: np.ndarray) -> dict:
    norms = np.linalg.norm(features, axis=1)
    return {
        "shape": list(features.shape),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "feature_mean": float(np.mean(features)),
        "feature_std": float(np.std(features)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned/random-feature MMD between reference and generated graphs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="RBF bandwidth. If omitted, median heuristic is used.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ref_graphs = _maybe_subsample(_load_reference_graphs(args.dataset), args.max_graphs, args.seed)
    gen_graphs = _maybe_subsample(_load_generated_graphs(args.dataset, args.model), args.max_graphs, args.seed + 1)

    logger.info("Evaluating learned-feature MMD: dataset=%s model=%s ref=%d gen=%d", args.dataset, args.model, len(ref_graphs), len(gen_graphs))

    encoder = RandomGINPlaceholder(feature_dim=args.feature_dim, seed=args.seed, normalize_output=True)
    ref_feats = _encode_graphs(ref_graphs, encoder)
    gen_feats = _encode_graphs(gen_graphs, encoder)
    score = feature_mmd(ref_feats, gen_feats, sigma=args.sigma)

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "learned_feature",
        "encoder": {
            "name": "RandomGINPlaceholder",
            "feature_dim": args.feature_dim,
            "seed": args.seed,
            "normalize_output": True,
            "note": "Fixed random-feature encoder; not a trained GNN.",
        },
        "num_reference_graphs": len(ref_graphs),
        "num_generated_graphs": len(gen_graphs),
        "kernel": {
            "type": "rbf",
            "sigma": args.sigma,
            "sigma_note": "None means median heuristic was used inside feature_mmd.",
        },
        "debug": {
            "reference_features": _feature_summary(ref_feats),
            "generated_features": _feature_summary(gen_feats),
        },
        "results": {"learned_feature_mmd": float(score)},
    }

    output_path = Path(args.output) if args.output else Path("outputs/metrics") / args.dataset / args.model / "learned_feature_metrics.json"
    save_json(payload, output_path)
    logger.info("Saved learned-feature metrics to %s. learned_feature_mmd=%.8f", output_path, score)


if __name__ == "__main__":
    main()
