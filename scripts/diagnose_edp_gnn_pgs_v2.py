#!/usr/bin/env python3
"""Diagnose high PolyGraphScore for EDP-GNN samples on survey datasets.

Reads persisted reference graphs, generated samples, PGS JSON, and metadata.
It does not recompute PGS. It reports simple topology statistics and SBM-specific
block statistics that usually explain why degree/spectral/clustering/GIN PGS is high.

Example:
  PYTHONPATH=src python scripts/diagnose_edp_gnn_pgs_v2.py \
    --dataset sbm --model edp_gnn --run-ids 0 1 2 \
    --max-reference-graphs 1024 --max-generated-graphs 1024
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def open_maybe_gzip(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def load_pickle(path: Path) -> Any:
    with open_maybe_gzip(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": repr(exc)}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_graphs(path: Path, split: str | None = None) -> list[nx.Graph]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_pickle(path)
    if isinstance(payload, Mapping):
        if split and split in payload:
            payload = payload[split]
        else:
            for key in ("graphs", "samples", "data", "generated_graphs", "reference_graphs"):
                if key in payload:
                    payload = payload[key]
                    break
    if hasattr(payload, "to_nx"):
        payload = payload.to_nx()
    graphs = list(payload) if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)) else [payload]
    out: list[nx.Graph] = []
    for graph in graphs:
        g = nx.Graph(graph)
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        g.remove_edges_from(nx.selfloop_edges(g))
        out.append(g)
    return out


def quantiles(x: Sequence[float]) -> dict[str, float]:
    if not x:
        return {}
    arr = np.asarray(x, dtype=float)
    return {
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def safe_average_clustering(g: nx.Graph) -> float:
    try:
        return float(nx.average_clustering(g))
    except Exception:
        return 0.0


def graph_stats(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    n = [g.number_of_nodes() for g in graphs]
    m = [g.number_of_edges() for g in graphs]
    density = [nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs]
    avg_degree = [(2.0 * g.number_of_edges() / max(g.number_of_nodes(), 1)) for g in graphs]
    isolates = [nx.number_of_isolates(g) for g in graphs]
    components, largest_component_frac, connected, planar, clustering, triangles = [], [], [], [], [], []
    for g in graphs:
        try:
            comps = list(nx.connected_components(g)) if g.number_of_nodes() else []
            components.append(len(comps))
            largest_component_frac.append(max((len(c) for c in comps), default=0) / max(g.number_of_nodes(), 1))
            connected.append(float(nx.is_connected(g)) if g.number_of_nodes() else 0.0)
        except Exception:
            components.append(float("nan")); largest_component_frac.append(float("nan")); connected.append(0.0)
        try:
            planar.append(float(nx.check_planarity(g)[0]))
        except Exception:
            planar.append(0.0)
        clustering.append(safe_average_clustering(g))
        try:
            triangles.append(float(sum(nx.triangles(g).values()) / 3.0))
        except Exception:
            triangles.append(0.0)
    return {
        "num_graphs": len(graphs),
        "num_nodes": quantiles(n),
        "num_edges": quantiles(m),
        "density": quantiles(density),
        "avg_degree": quantiles(avg_degree),
        "isolates": quantiles(isolates),
        "components": quantiles(components),
        "largest_component_fraction": quantiles(largest_component_frac),
        "avg_clustering": quantiles(clustering),
        "triangle_count": quantiles(triangles),
        "connected_rate": float(np.mean(connected)) if connected else None,
        "planarity_rate": float(np.mean(planar)) if planar else None,
    }


def ks_distance(a: Sequence[float], b: Sequence[float]) -> float | None:
    if not a or not b:
        return None
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        x = np.sort(np.unique(np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])))
        aa = np.sort(np.asarray(a, dtype=float)); bb = np.sort(np.asarray(b, dtype=float))
        fa = np.searchsorted(aa, x, side="right") / len(aa)
        fb = np.searchsorted(bb, x, side="right") / len(bb)
        return float(np.max(np.abs(fa - fb)))


def vector_values(graphs: Sequence[nx.Graph], name: str) -> list[float]:
    if name == "edges":
        return [float(g.number_of_edges()) for g in graphs]
    if name == "nodes":
        return [float(g.number_of_nodes()) for g in graphs]
    if name == "density":
        return [float(nx.density(g) if g.number_of_nodes() > 1 else 0.0) for g in graphs]
    if name == "avg_degree":
        return [float(2 * g.number_of_edges() / max(g.number_of_nodes(), 1)) for g in graphs]
    if name == "isolates":
        return [float(nx.number_of_isolates(g)) for g in graphs]
    if name == "avg_clustering":
        return [safe_average_clustering(g) for g in graphs]
    if name == "triangles":
        vals = []
        for g in graphs:
            try:
                vals.append(float(sum(nx.triangles(g).values()) / 3.0))
            except Exception:
                vals.append(0.0)
        return vals
    raise KeyError(name)


def sample_path(root: Path, dataset: str, model: str, run_id: int | None) -> Path:
    if run_id is None:
        return root / "outputs" / "samples" / dataset / f"{model}.pkl"
    return root / "outputs" / "samples" / dataset / model / f"run_{run_id:03d}.pkl"


def metric_path(root: Path, dataset: str, model: str, run_id: int | None, filename: str) -> Path:
    if run_id is None:
        return root / "outputs" / "metrics" / dataset / model / filename
    return root / "outputs" / "metrics" / dataset / model / f"run_{run_id:03d}" / filename


def metadata_paths(root: Path, dataset: str, model: str, run_id: int | None) -> dict[str, Path]:
    if run_id is None:
        return {
            "sample_metadata": root / "outputs" / "samples" / dataset / f"{model}.metadata.json",
            "train_metadata": root / "outputs" / "runs" / dataset / model / "train_metadata.json",
            "resolved_model_config": root / "outputs" / "runs" / dataset / model / "resolved_model_config.yaml",
            "sample_config": root / "outputs" / "samples" / dataset / f"{model}.resolved_model_config.yaml",
        }
    return {
        "sample_metadata": root / "outputs" / "samples" / dataset / model / f"run_{run_id:03d}.metadata.json",
        "train_metadata": root / "outputs" / "runs" / dataset / model / f"run_{run_id:03d}" / "train_metadata.json",
        "resolved_model_config": root / "outputs" / "runs" / dataset / model / f"run_{run_id:03d}" / "resolved_model_config.yaml",
        "sample_config": root / "outputs" / "samples" / dataset / model / f"run_{run_id:03d}.resolved_model_config.yaml",
    }


def subsample(graphs: list[nx.Graph], n: int | None, seed: int) -> list[nx.Graph]:
    if n is None or n <= 0 or len(graphs) <= n:
        return graphs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=n, replace=False)
    return [graphs[int(i)] for i in idx]


def sbm_config(root: Path, dataset: str) -> dict[str, Any]:
    cfg = load_yaml(root / "configs" / "datasets" / f"{dataset}.yaml")
    if str(cfg.get("name", dataset)).lower() != "sbm" and dataset.lower() != "sbm":
        return {}
    n = int(cfg.get("num_nodes", 64))
    b = int(cfg.get("num_blocks", 4))
    base = n // b
    sizes = [base] * b
    sizes[-1] += n - sum(sizes)
    p_in = float(cfg.get("p_in", 0.25))
    p_out = float(cfg.get("p_out", 0.02))
    within_possible = sum(s * (s - 1) / 2 for s in sizes)
    total_possible = n * (n - 1) / 2
    between_possible = total_possible - within_possible
    return {
        "num_nodes": n,
        "num_blocks": b,
        "block_sizes": sizes,
        "p_in": p_in,
        "p_out": p_out,
        "expected_within_edges": within_possible * p_in,
        "expected_between_edges": between_possible * p_out,
        "expected_total_edges": within_possible * p_in + between_possible * p_out,
        "expected_density": (within_possible * p_in + between_possible * p_out) / total_possible,
        "expected_avg_degree": 2 * (within_possible * p_in + between_possible * p_out) / n,
    }


def sbm_block_stats(graphs: Sequence[nx.Graph], cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg:
        return {}
    sizes = [int(x) for x in cfg["block_sizes"]]
    starts = np.cumsum([0] + sizes)
    node_to_block = {}
    for block, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        for node in range(int(lo), int(hi)):
            node_to_block[node] = block
    within, between, ratio, missing_nodes = [], [], [], []
    for g in graphs:
        w = 0; b = 0
        for u, v in g.edges():
            bu = node_to_block.get(int(u)); bv = node_to_block.get(int(v))
            if bu is None or bv is None:
                continue
            if bu == bv:
                w += 1
            else:
                b += 1
        within.append(float(w)); between.append(float(b)); ratio.append(float(w / max(w + b, 1)))
        missing_nodes.append(max(0, int(cfg["num_nodes"]) - g.number_of_nodes()))
    return {
        "within_edges": quantiles(within),
        "between_edges": quantiles(between),
        "within_edge_fraction": quantiles(ratio),
        "missing_nodes_vs_config": quantiles(missing_nodes),
    }


def expected_uncorrected_noise_edges(n_mean: float) -> float:
    possible_edges = n_mean * max(n_mean - 1, 0) / 2.0
    # P(|N(0,1)| >= 0.5)
    p_abs_normal_ge_05 = math.erfc(0.5 / math.sqrt(2.0))
    return possible_edges * p_abs_normal_ge_05


def diagnose_run(args: argparse.Namespace, run_id: int | None) -> dict[str, Any]:
    root = Path(args.survey_root).resolve()
    ref_path = root / args.dataset_root / args.dataset / f"{args.reference_split}.pkl"
    gen_path = sample_path(root, args.dataset, args.model, run_id)
    ref_graphs = subsample(load_graphs(ref_path), args.max_reference_graphs, args.seed + (run_id or 0) * 1000)
    gen_graphs = subsample(load_graphs(gen_path), args.max_generated_graphs, args.seed + (run_id or 0) * 1000 + 1)
    n_pairs = min(len(ref_graphs), len(gen_graphs))
    ref_graphs = ref_graphs[:n_pairs]
    gen_graphs = gen_graphs[:n_pairs]

    ref_stats = graph_stats(ref_graphs)
    gen_stats = graph_stats(gen_graphs)
    comparisons = {}
    for key in ["nodes", "edges", "density", "avg_degree", "isolates", "avg_clustering", "triangles"]:
        rv = vector_values(ref_graphs, key); gv = vector_values(gen_graphs, key)
        rmean = float(np.mean(rv)) if rv else float("nan")
        gmean = float(np.mean(gv)) if gv else float("nan")
        comparisons[key] = {
            "reference_mean": rmean,
            "generated_mean": gmean,
            "mean_difference_generated_minus_reference": gmean - rmean,
            "generated_over_reference_mean_ratio": (gmean / rmean) if abs(rmean) > 1e-12 else None,
            "ks_distance": ks_distance(rv, gv),
        }

    metric = load_json(metric_path(root, args.dataset, args.model, run_id, args.metric_filename))
    paths = metadata_paths(root, args.dataset, args.model, run_id)
    sample_meta = load_json(paths["sample_metadata"])
    train_meta = load_json(paths["train_metadata"])
    train_cfg = load_yaml(paths["resolved_model_config"])
    sample_cfg = load_yaml(paths["sample_config"])
    dcfg = sbm_config(root, args.dataset)

    n_mean = ref_stats.get("num_nodes", {}).get("mean", 0.0) or 0.0
    flags = []
    edge_ratio = comparisons["edges"]["generated_over_reference_mean_ratio"]
    if edge_ratio is not None and (edge_ratio > 2.0 or edge_ratio < 0.5):
        flags.append("large_edge_count_mismatch_degree_descriptor_can_separate")
    if comparisons["density"]["ks_distance"] is not None and comparisons["density"]["ks_distance"] > 0.5:
        flags.append("large_density_distribution_mismatch")
    if comparisons["avg_clustering"]["ks_distance"] is not None and comparisons["avg_clustering"]["ks_distance"] > 0.5:
        flags.append("large_clustering_distribution_mismatch")
    if comparisons["isolates"]["generated_mean"] > comparisons["isolates"]["reference_mean"] + 1.0:
        flags.append("generated_samples_have_extra_isolates")
    if ref_stats.get("planarity_rate") is not None and gen_stats.get("planarity_rate") is not None:
        if float(ref_stats["planarity_rate"]) - float(gen_stats["planarity_rate"]) > 0.2:
            flags.append("generated_graphs_often_non_planar")

    sbm_ref = sbm_block_stats(ref_graphs, dcfg)
    sbm_gen = sbm_block_stats(gen_graphs, dcfg)
    if sbm_ref and sbm_gen:
        ref_frac = sbm_ref["within_edge_fraction"].get("mean")
        gen_frac = sbm_gen["within_edge_fraction"].get("mean")
        if ref_frac is not None and gen_frac is not None and abs(float(ref_frac) - float(gen_frac)) > 0.15:
            flags.append("sbm_block_structure_mismatch_spectral_gin_can_separate")

    pgs_results = metric.get("results", {})
    return {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": run_id,
        "reference_path": str(ref_path),
        "generated_path": str(gen_path),
        "num_pairs_used": n_pairs,
        "pgs_results": pgs_results,
        "pgs_protocol": metric.get("protocol", {}),
        "reference_stats": ref_stats,
        "generated_stats": gen_stats,
        "comparisons": comparisons,
        "sbm_config": dcfg,
        "reference_sbm_block_stats": sbm_ref,
        "generated_sbm_block_stats": sbm_gen,
        "sample_metadata_quality": sample_meta.get("quality", {}),
        "train_metadata": {
            "checkpoint_path": train_meta.get("checkpoint_path"),
            "training_time_seconds": train_meta.get("training_time_seconds"),
            "split_sizes": train_meta.get("split_sizes"),
            "model_config_hash": train_meta.get("model_config_hash"),
        },
        "resolved_edp_config": {
            "num_epochs": train_cfg.get("num_epochs") or train_cfg.get("train", {}).get("max_epoch"),
            "eps": train_cfg.get("eps") or train_cfg.get("mcmc", {}).get("eps") or sample_cfg.get("eps") or sample_cfg.get("mcmc", {}).get("eps"),
            "grad_step_size": train_cfg.get("grad_step_size") or train_cfg.get("mcmc", {}).get("grad_step_size") or sample_cfg.get("grad_step_size") or sample_cfg.get("mcmc", {}).get("grad_step_size"),
            "step_num": train_cfg.get("step_num") or train_cfg.get("mcmc", {}).get("step_num"),
            "sigmas": train_cfg.get("sigmas") or train_cfg.get("train", {}).get("sigmas"),
            "checkpoint_path": train_cfg.get("checkpoint_path") or sample_cfg.get("checkpoint_path"),
        },
        "edp_noise_baseline": {
            "note": "EDP-GNN gen_init_sample uses abs(N(0,1)); rounding at 0.5 gives this expected edge count if sampling stays close to initialization.",
            "reference_num_nodes_mean": n_mean,
            "p_abs_normal_ge_0_5": math.erfc(0.5 / math.sqrt(2.0)),
            "expected_edges_after_threshold_if_uncorrected_noise": expected_uncorrected_noise_edges(n_mean),
        },
        "flags": flags,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--survey-root", default=".")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="edp_gnn")
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--run-ids", type=int, nargs="*", default=None)
    p.add_argument("--dataset-root", default="outputs/datasets")
    p.add_argument("--reference-split", default="test")
    p.add_argument("--metric-filename", default="polygraphscore_official.json")
    p.add_argument("--max-reference-graphs", type=int, default=None)
    p.add_argument("--max-generated-graphs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.run_id is not None and args.run_ids:
        raise SystemExit("Use either --run-id or --run-ids, not both")
    if args.run_id is not None:
        run_ids: list[int | None] = [args.run_id]
    elif args.run_ids:
        run_ids = list(args.run_ids)
    else:
        run_ids = [None]

    records = [diagnose_run(args, rid) for rid in run_ids]
    payload: dict[str, Any] = {"records": records}
    if len(records) > 1:
        def mean_or_nan(path):
            vals = []
            for r in records:
                cur = r
                try:
                    for key in path:
                        cur = cur[key]
                    vals.append(float(cur))
                except Exception:
                    pass
            return float(np.mean(vals)) if vals else None
        payload["summary"] = {
            "pgs_mean": mean_or_nan(["pgs_results", "pgs"]),
            "edge_ratio_mean": mean_or_nan(["comparisons", "edges", "generated_over_reference_mean_ratio"]),
            "density_ratio_mean": mean_or_nan(["comparisons", "density", "generated_over_reference_mean_ratio"]),
            "generated_planarity_mean": mean_or_nan(["generated_stats", "planarity_rate"]),
            "sbm_within_fraction_ref_mean": mean_or_nan(["reference_sbm_block_stats", "within_edge_fraction", "mean"]),
            "sbm_within_fraction_gen_mean": mean_or_nan(["generated_sbm_block_stats", "within_edge_fraction", "mean"]),
            "flags": sorted(set(flag for r in records for flag in r.get("flags", []))),
        }

    out = Path(args.output) if args.output else Path("outputs/metrics") / args.dataset / args.model / "edp_gnn_pgs_diagnostics_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for r in records:
        pgs = r.get("pgs_results", {}).get("pgs")
        desc = r.get("pgs_results", {}).get("pgs_descriptor")
        ref_e = r["reference_stats"]["num_edges"].get("mean", float("nan"))
        gen_e = r["generated_stats"]["num_edges"].get("mean", float("nan"))
        ref_d = r["reference_stats"]["density"].get("mean", float("nan"))
        gen_d = r["generated_stats"]["density"].get("mean", float("nan"))
        flags = ", ".join(r["flags"]) or "none"
        print(f"run={r['run_id']} pgs={pgs} descriptor={desc} ref_edges_mean={ref_e:.2f} gen_edges_mean={gen_e:.2f} ref_density={ref_d:.3f} gen_density={gen_d:.3f} flags={flags}")
        if r.get("sbm_config"):
            ref_frac = r.get("reference_sbm_block_stats", {}).get("within_edge_fraction", {}).get("mean")
            gen_frac = r.get("generated_sbm_block_stats", {}).get("within_edge_fraction", {}).get("mean")
            print(f"  sbm expected_edges={r['sbm_config'].get('expected_total_edges'):.2f} ref_within_frac={ref_frac} gen_within_frac={gen_frac}")
        print(f"  uncorrected_noise_expected_edges={r['edp_noise_baseline']['expected_edges_after_threshold_if_uncorrected_noise']:.2f}")
    print(f"Saved diagnostics to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
