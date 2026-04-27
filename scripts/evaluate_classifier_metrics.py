from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.metrics.classifier.dataset import make_binary_dataset
from empirical_comparison.metrics.classifier.features import GraphDescriptorFeaturizer
from empirical_comparison.metrics.classifier.score import classifier_scores
from empirical_comparison.metrics.classifier.train import PolyGraphScoreClassifier
from empirical_comparison.registry import DATASET_REGISTRY
from empirical_comparison.utils.io import load_yaml, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _load_reference_graphs(dataset: str) -> list:
    data_cfg_path = Path("configs/datasets") / f"{dataset}.yaml"
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_cfg_path}")

    data_cfg = load_yaml(data_cfg_path)

    if dataset not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{dataset}' is not registered. "
            f"Available datasets: {sorted(DATASET_REGISTRY.keys())}"
        )

    splits = DATASET_REGISTRY[dataset](data_cfg).build()
    if "test" not in splits:
        raise KeyError(f"Dataset '{dataset}' did not return a test split.")

    graphs = list(splits["test"])
    if len(graphs) == 0:
        raise ValueError(f"Test split for dataset '{dataset}' is empty.")

    return graphs


def _load_generated_graphs(dataset: str, model: str) -> list:
    sample_path = Path("outputs/samples") / dataset / f"{model}.pkl"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Generated sample file not found: {sample_path}. "
            f"Run generate_samples.py first."
        )

    with open(sample_path, "rb") as f:
        graphs = pickle.load(f)

    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs as list, got {type(graphs)}.")

    if len(graphs) == 0:
        raise ValueError(f"No generated graphs found in {sample_path}.")

    return graphs


def _subsample(graphs: list, max_graphs: int | None, seed: int) -> list:
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classifier-based graph generation metrics."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-k", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ref_graphs = _load_reference_graphs(args.dataset)
    gen_graphs = _load_generated_graphs(args.dataset, args.model)

    ref_graphs = _subsample(ref_graphs, args.max_graphs, args.seed)
    gen_graphs = _subsample(gen_graphs, args.max_graphs, args.seed + 1)

    logger.info(
        "Evaluating classifier metric: dataset=%s model=%s ref=%d gen=%d",
        args.dataset,
        args.model,
        len(ref_graphs),
        len(gen_graphs),
    )

    featurizer = GraphDescriptorFeaturizer(
        degree_bins=args.degree_bins,
        clustering_bins=args.clustering_bins,
        spectral_k=args.spectral_k,
        max_degree=args.max_degree,
    )

    ref_features = featurizer.transform(ref_graphs)
    gen_features = featurizer.transform(gen_graphs)

    x, y = make_binary_dataset(ref_features, gen_features)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.test_size,
        random_state=args.seed,
    )

    split_results: list[dict[str, float]] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        clf = PolyGraphScoreClassifier(random_state=args.seed + split_id)
        clf.fit(x_train, y_train)

        prob_generated = clf.predict_proba_generated(x_test)
        scores = classifier_scores(prob_generated, y_test)
        split_results.append(scores)

        logger.info(
            "split=%d auc=%.4f pgs_js_normalized=%.4f acc=%.4f",
            split_id,
            scores["classifier_auc"],
            scores["pgs_js_normalized"],
            scores["classifier_accuracy"],
        )

    metric_names = split_results[0].keys()
    results = {}
    for name in metric_names:
        values = np.asarray([r[name] for r in split_results], dtype=np.float64)
        results[f"{name}_mean"] = float(values.mean())
        results[f"{name}_std"] = float(values.std(ddof=0))

    output_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "classifier_based",
        "feature_representation": {
            "name": "GraphDescriptorFeaturizer",
            "degree_bins": args.degree_bins,
            "clustering_bins": args.clustering_bins,
            "spectral_k": args.spectral_k,
            "max_degree": args.max_degree,
        },
        "classifier": {
            "name": "PolyGraphScoreClassifier",
            "base_model": "standardized_logistic_regression",
            "note": "PGS-style descriptor classifier; not the full TabPFN PolyGraphScore implementation.",
        },
        "protocol": {
            "num_reference_graphs": len(ref_graphs),
            "num_generated_graphs": len(gen_graphs),
            "test_size": args.test_size,
            "num_splits": args.num_splits,
            "seed": args.seed,
            "split": "stratified_shuffle_split",
        },
        "interpretation": {
            "classifier_auc": "0.5 is ideal / indistinguishable; 1.0 is highly separable.",
            "pgs_js_normalized": "0 is low discrepancy; 1 is high discrepancy.",
        },
        "results": results,
        "split_results": split_results,
    }

    if args.output is None:
        output_path = (
            Path("outputs/metrics")
            / args.dataset
            / args.model
            / "classifier_metrics.json"
        )
    else:
        output_path = Path(args.output)

    save_json(output_payload, output_path)

    logger.info("Saved classifier metrics to %s", output_path)


if __name__ == "__main__":
    main()