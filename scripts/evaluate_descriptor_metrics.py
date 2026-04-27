from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Callable, Sequence

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.metrics.descriptor.descriptors import (
    clustering_histogram,
    degree_histogram,
    orbit_count_vector,
    spectral_histogram,
    structural_summary,
)
from empirical_comparison.metrics.descriptor.mmd import mmd_gaussian_emd, mmd_unbiased
from empirical_comparison.registry import DATASET_REGISTRY
from empirical_comparison.utils.io import load_yaml, save_json
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _load_reference_graphs(dataset: str) -> list[nx.Graph]:
    data_cfg_path = Path("configs/datasets") / f"{dataset}.yaml"
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_cfg_path}")
    data_cfg = load_yaml(data_cfg_path)
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset}' is not registered. Available: {sorted(DATASET_REGISTRY.keys())}")
    splits = DATASET_REGISTRY[dataset](data_cfg).build()
    if "test" not in splits:
        raise KeyError(f"Dataset '{dataset}' did not return a test split.")
    graphs = list(splits["test"])
    if len(graphs) == 0:
        raise ValueError(f"Test split for dataset '{dataset}' is empty.")
    return graphs


def _load_generated_graphs(dataset: str, model: str) -> list[nx.Graph]:
    sample_path = Path("outputs/samples") / dataset / f"{model}.pkl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_path}. Run generate_samples.py first.")
    with open(sample_path, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected generated graphs to be a list, got {type(graphs)}")
    if len(graphs) == 0:
        raise ValueError(f"No generated graphs found in {sample_path}.")
    return graphs


def _subsample(graphs: Sequence[nx.Graph], max_graphs: int | None, seed: int) -> list[nx.Graph]:
    graphs = list(graphs)
    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return graphs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=max_graphs, replace=False)
    return [graphs[i] for i in idx]


def _descriptor_matrix(graphs: Sequence[nx.Graph], fn: Callable[[nx.Graph], np.ndarray]) -> np.ndarray:
    x = np.asarray([fn(g) for g in graphs], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Descriptor function returned invalid shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Descriptor matrix contains NaN or Inf values.")
    return x


def _mmd_with_optional_bootstrap(
    ref_desc: np.ndarray,
    gen_desc: np.ndarray,
    *,
    metric_kind: str,
    sigma: float | None,
    num_bootstrap: int,
    seed: int,
) -> tuple[float, float | None]:
    """Compute MMD and optional bootstrap std.

    metric_kind:
        - "rbf": RBF-kernel MMD with median heuristic if sigma is None.
        - "emd": Gaussian-EMD-kernel MMD, matching common graph-generation
          histogram MMD code. If sigma is None, defaults to 1.0.
    """
    if metric_kind == "rbf":
        compute = lambda a, b: mmd_unbiased(a, b, sigma=sigma)
    elif metric_kind == "emd":
        sigma_value = 1.0 if sigma is None else sigma
        compute = lambda a, b: mmd_gaussian_emd(a, b, sigma=sigma_value, unbiased=False)
    else:
        raise ValueError(f"Unknown metric_kind: {metric_kind}")

    base = compute(ref_desc, gen_desc)
    if num_bootstrap <= 0:
        return base, None

    rng = np.random.default_rng(seed)
    n = min(len(ref_desc), len(gen_desc))
    vals: list[float] = []
    for _ in range(num_bootstrap):
        ref_idx = rng.choice(len(ref_desc), size=n, replace=True)
        gen_idx = rng.choice(len(gen_desc), size=n, replace=True)
        vals.append(compute(ref_desc[ref_idx], gen_desc[gen_idx]))
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate descriptor-based graph generation metrics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="Kernel bandwidth. For EMD-kernel metrics, default is 1.0. For RBF metrics, None uses median heuristic.")
    parser.add_argument("--num-bootstrap", type=int, default=0, help="Number of bootstrap rounds for std estimates.")
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--clustering-bins", type=int, default=20)
    parser.add_argument("--spectral-bins", type=int, default=20)
    parser.add_argument("--max-degree", type=int, default=100)
    parser.add_argument("--orca-exec", type=str, default=None, help="Path to ORCA executable. Overrides ORCA_EXEC environment variable.")
    parser.add_argument("--skip-orbit", action="store_true", help="Skip ORCA orbit MMD and use structural summary only.")
    parser.add_argument("--orbit-log-transform", action="store_true", help="Apply log1p to ORCA orbit counts before normalization.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ref_graphs = _subsample(_load_reference_graphs(args.dataset), args.max_graphs, args.seed)
    gen_graphs = _subsample(_load_generated_graphs(args.dataset, args.model), args.max_graphs, args.seed + 1)

    logger.info(
        "Evaluating descriptor metrics: dataset=%s model=%s ref=%d gen=%d",
        args.dataset,
        args.model,
        len(ref_graphs),
        len(gen_graphs),
    )

    descriptor_specs: dict[str, tuple[Callable[[nx.Graph], np.ndarray], str]] = {
        # EMD-kernel MMD is common for histogram descriptors in graph-generation evaluation.
        "degree_mmd": (
            lambda g: degree_histogram(g, bins=args.degree_bins, max_degree=args.max_degree),
            "emd",
        ),
        "clustering_mmd": (
            lambda g: clustering_histogram(g, bins=args.clustering_bins),
            "emd",
        ),
        "spectral_mmd": (
            lambda g: spectral_histogram(g, bins=args.spectral_bins),
            "emd",
        ),
        # Lightweight non-ORCA summary retained for debugging and fallback tables.
        "structural_summary_mmd": (structural_summary, "rbf"),
    }

    if not args.skip_orbit:
        descriptor_specs["orbit_mmd"] = (
            lambda g: orbit_count_vector(
                g,
                orca_exec=args.orca_exec,
                normalize=True,
                log_transform=args.orbit_log_transform,
            ),
            "emd",
        )

    results: dict[str, float] = {}
    debug: dict[str, dict] = {}
    for name, (fn, metric_kind) in descriptor_specs.items():
        logger.info("Computing %s", name)
        try:
            ref_desc = _descriptor_matrix(ref_graphs, fn)
            gen_desc = _descriptor_matrix(gen_graphs, fn)
        except FileNotFoundError as exc:
            if name == "orbit_mmd":
                raise FileNotFoundError(
                    "orbit_mmd requires ORCA. Install/compile ORCA and set ORCA_EXEC, "
                    "or rerun with --skip-orbit."
                ) from exc
            raise

        mean, std = _mmd_with_optional_bootstrap(
            ref_desc,
            gen_desc,
            metric_kind=metric_kind,
            sigma=args.sigma,
            num_bootstrap=args.num_bootstrap,
            seed=args.seed,
        )
        results[name] = mean
        if std is not None:
            results[f"{name}_std"] = std
        debug[name] = {
            "reference_shape": list(ref_desc.shape),
            "generated_shape": list(gen_desc.shape),
            "reference_mean_norm": float(np.linalg.norm(ref_desc, axis=1).mean()),
            "generated_mean_norm": float(np.linalg.norm(gen_desc, axis=1).mean()),
            "mmd_kernel": "gaussian_emd" if metric_kind == "emd" else "rbf",
        }

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "metric_family": "descriptor_based",
        "num_reference_graphs": len(ref_graphs),
        "num_generated_graphs": len(gen_graphs),
        "protocol": {
            "seed": args.seed,
            "max_graphs": args.max_graphs,
            "sigma": args.sigma,
            "sigma_note": "For EMD-kernel MMD, None uses sigma=1.0. For RBF MMD, None uses median heuristic.",
            "num_bootstrap": args.num_bootstrap,
            "degree_bins": args.degree_bins,
            "clustering_bins": args.clustering_bins,
            "spectral_bins": args.spectral_bins,
            "max_degree": args.max_degree,
            "orca_exec": args.orca_exec,
            "skip_orbit": args.skip_orbit,
            "orbit_log_transform": args.orbit_log_transform,
        },
        "notes": {
            "orbit_mmd": "True ORCA-based 4-node orbit-count MMD when --skip-orbit is false and ORCA is configured.",
            "structural_summary_mmd": "Lightweight structural-summary fallback, not a graphlet/orbit metric.",
        },
        "debug": debug,
        "results": results,
    }

    output_path = Path(args.output) if args.output else Path("outputs/metrics") / args.dataset / args.model / "descriptor_metrics.json"
    save_json(payload, output_path)
    logger.info("Saved descriptor metrics to %s", output_path)


if __name__ == "__main__":
    main()
