from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.data_io import load_dataset_splits
from empirical_comparison.evaluation.run_utils import (
    make_model_config,
    make_model_run_config,
    sample_config_path,
    sample_metadata_path,
    sample_path,
)
from empirical_comparison.generation.sampler import model_capabilities, sample_graphs
from empirical_comparison.graphs.attributes import apply_empirical_attributes, attribute_coverage, fit_attribute_statistics, normalize_schema
from empirical_comparison.generation.validity import quality_metrics
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.compute import PeakMemoryMonitor, compute_report
from empirical_comparison.utils.io import load_yaml, save_json, save_pickle, save_yaml, stable_hash
from empirical_comparison.utils.logging import get_logger
from empirical_comparison.utils.numerics import assert_finite_graphs
from empirical_comparison.utils.seed import set_seed

logger = get_logger(__name__)


def _cpu_requested(config: dict) -> bool:
    device = str(config.get("device", "")).lower()
    if device == "cpu":
        return True
    try:
        return int(config.get("gpus", 1)) == 0
    except Exception:
        return False


def _hide_cuda_for_cpu_config(config: dict) -> None:
    if _cpu_requested(config):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU model config detected; set CUDA_VISIBLE_DEVICES='' before importing model wrapper.")


def _label_value(graph: nx.Graph, node: int, attr: str):
    if node not in graph:
        return None
    return graph.nodes[node].get(attr)


def _edge_value(graph: nx.Graph, u: int, v: int, attr: str):
    if not graph.has_edge(u, v):
        return None
    return graph.edges[u, v].get(attr)


def _union_layout(source: nx.Graph, target: nx.Graph, *, seed: int) -> dict[int, np.ndarray]:
    union = nx.Graph()
    n = max(source.number_of_nodes(), target.number_of_nodes())
    union.add_nodes_from(range(n))
    union.add_edges_from((int(u), int(v)) for u, v in source.edges() if int(u) < n and int(v) < n)
    union.add_edges_from((int(u), int(v)) for u, v in target.edges() if int(u) < n and int(v) < n)
    if union.number_of_nodes() == 0:
        return {}
    if union.number_of_nodes() == 1:
        return {next(iter(union.nodes())): np.asarray([0.0, 0.0])}
    return nx.spring_layout(union, seed=seed)


def _trajectory_state(
    source: nx.Graph,
    target: nx.Graph,
    *,
    alpha: float,
    node_label_attr: str,
    edge_label_attr: str,
) -> nx.Graph:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n_source = source.number_of_nodes()
    n_target = target.number_of_nodes()
    n = int(round((1.0 - alpha) * n_source + alpha * n_target))
    n = max(1, min(max(n_source, n_target), n))
    g = nx.Graph()
    g.add_nodes_from(range(n))

    for node in range(n):
        source_label = _label_value(source, node, node_label_attr)
        target_label = _label_value(target, node, node_label_attr)
        label = target_label if alpha >= 0.5 and target_label is not None else source_label
        if label is not None:
            g.nodes[node][node_label_attr] = label

    source_edges = {tuple(sorted((int(u), int(v)))) for u, v in source.edges() if int(u) < n and int(v) < n and u != v}
    target_edges = {tuple(sorted((int(u), int(v)))) for u, v in target.edges() if int(u) < n and int(v) < n and u != v}
    kept_source = sorted(source_edges - target_edges)
    added_target = sorted(target_edges - source_edges)
    shared = sorted(source_edges & target_edges)
    drop_count = int(math.floor(alpha * len(kept_source)))
    add_count = int(math.ceil(alpha * len(added_target)))
    state_edges = set(shared)
    state_edges.update(kept_source[drop_count:])
    state_edges.update(added_target[:add_count])

    for u, v in sorted(state_edges):
        source_type = _edge_value(source, u, v, edge_label_attr)
        target_type = _edge_value(target, u, v, edge_label_attr)
        edge_type = target_type if alpha >= 0.5 and target_type is not None else source_type
        attrs = {}
        if edge_type is not None:
            attrs[edge_label_attr] = edge_type
        g.add_edge(u, v, **attrs)
    return g


def _draw_graph_state(
    ax,
    graph: nx.Graph,
    *,
    pos: dict[int, np.ndarray],
    title: str,
    node_label_attr: str,
    edge_label_attr: str,
    show_node_labels: bool,
    show_edge_labels: bool,
) -> None:
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return
    labels = [graph.nodes[n].get(node_label_attr, graph.degree(n)) for n in graph.nodes()]
    label_ids = {str(v): i for i, v in enumerate(sorted({str(v) for v in labels}))}
    colors = [label_ids[str(v)] for v in labels]
    node_size = max(80, min(260, int(1800 / max(graph.number_of_nodes(), 1))))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#8a8f98", width=1.0, alpha=0.75)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        cmap="tab20",
        node_size=node_size,
        linewidths=0.5,
        edgecolors="#222222",
    )
    if show_node_labels:
        node_labels = {n: str(graph.nodes[n].get(node_label_attr, n)) for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=7)
    if show_edge_labels:
        edge_labels = {(u, v): str(data.get(edge_label_attr, "")) for u, v, data in graph.edges(data=True)}
        edge_labels = {edge: label for edge, label in edge_labels.items() if label}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=6)


def _draw_sampling_trajectory(
    *,
    dataset: str,
    model_name: str,
    reference_graphs: list[nx.Graph],
    generated_graphs: list[nx.Graph],
    output_path: Path,
    num_graphs: int,
    steps: int,
    seed: int,
    node_label_attr: str,
    edge_label_attr: str,
    show_node_labels: bool,
    show_edge_labels: bool,
    dpi: int,
) -> dict:
    if not reference_graphs:
        raise ValueError("Cannot draw trajectory because no reference graphs are available.")
    if not generated_graphs:
        raise ValueError("Cannot draw trajectory because no generated graphs are available.")
    num_graphs = max(1, min(int(num_graphs), len(generated_graphs)))
    steps = max(2, int(steps))
    rng = np.random.default_rng(seed)
    ref_indices = rng.choice(len(reference_graphs), size=num_graphs, replace=len(reference_graphs) < num_graphs)
    gen_indices = np.arange(num_graphs)

    fig, axes = plt.subplots(num_graphs, steps, figsize=(2.4 * steps, 2.3 * num_graphs), squeeze=False)
    alphas = np.linspace(0.0, 1.0, steps)
    for row, (ref_idx, gen_idx) in enumerate(zip(ref_indices, gen_indices)):
        source = nx.convert_node_labels_to_integers(nx.Graph(reference_graphs[int(ref_idx)]), ordering="sorted")
        target = nx.convert_node_labels_to_integers(nx.Graph(generated_graphs[int(gen_idx)]), ordering="sorted")
        pos = _union_layout(source, target, seed=seed + row)
        for col, alpha in enumerate(alphas):
            state = _trajectory_state(source, target, alpha=float(alpha), node_label_attr=node_label_attr, edge_label_attr=edge_label_attr)
            title = "ref" if col == 0 else ("sample" if col == steps - 1 else f"t={alpha:.2f}")
            _draw_graph_state(
                axes[row][col],
                state,
                pos=pos,
                title=title,
                node_label_attr=node_label_attr,
                edge_label_attr=edge_label_attr,
                show_node_labels=show_node_labels,
                show_edge_labels=show_edge_labels,
            )
    fig.suptitle(f"{dataset}/{model_name} reference-to-sample trajectory", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output_path),
        "num_trajectories": num_graphs,
        "steps": steps,
        "type": "reference_to_sample_visualization",
        "note": "This is a benchmark-level visualization from a reference graph to a generated graph. It is not an upstream model's internal denoising path unless the wrapper explicitly exposes one.",
    }


def _generate_samples(
    *,
    model_name: str,
    dataset: str,
    base_cfg: dict,
    cfg_path: Path,
    dataset_root: str,
    num_samples: int,
    seed: int,
    sample_seed_offset: int,
    force: bool,
    dry_run: bool,
    run_id: int | None = None,
    use_run_paths: bool = False,
    show_progress: bool,
    draw_trajectory: bool,
    trajectory_graphs: int,
    trajectory_steps: int,
    trajectory_output: Path | None,
    trajectory_node_label_attr: str,
    trajectory_edge_label_attr: str,
    trajectory_show_node_labels: bool,
    trajectory_show_edge_labels: bool,
    trajectory_dpi: int,
) -> dict:
    seed = int(seed) + int(sample_seed_offset)
    set_seed(seed)
    if run_id is None and not use_run_paths:
        cfg = make_model_config(
            base_cfg,
            dataset=dataset,
            model=model_name,
            seed=seed,
        )
    else:
        cfg = make_model_run_config(
            base_cfg,
            dataset=dataset,
            model=model_name,
            run_id=run_id,
            seed=seed,
            use_run_paths=use_run_paths,
        )
    cfg["num_samples"] = int(num_samples)

    path_run_id = run_id if use_run_paths else None
    out = sample_path(dataset, model_name, run_id=path_run_id)
    metadata_out = sample_metadata_path(dataset, model_name, run_id=path_run_id)
    resolved_cfg_out = sample_config_path(dataset, model_name, run_id=path_run_id)
    logger.info(
        "Generating samples: dataset=%s model=%s num_samples=%d seed=%d checkpoint=%s",
        dataset,
        model_name,
        num_samples,
        seed,
        cfg.get("checkpoint_path"),
    )
    if dry_run:
        logger.info("Dry run: would write %s", out)
        if draw_trajectory:
            logger.info("Dry run: would draw sampling trajectory")
        return {"dataset": dataset, "model": model_name, "seed": seed, "run_id": run_id, "sample_path": str(out), "dry_run": True}
    if out.exists() and not force:
        raise FileExistsError(f"Sample file already exists: {out}. Use --force to overwrite.")

    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        graphs = sample_graphs(
            model_name,
            cfg,
            num_samples,
            seed=seed,
            show_progress=show_progress,
            progress_desc=f"Sampling {dataset}/{model_name}" + (f"/run_{run_id:03d}" if run_id is not None else ""),
        )
        assert_finite_graphs(graphs, context=f"sampling {dataset}/{model_name} seed={seed}")
    elapsed = time.perf_counter() - start
    compute = compute_report(
        operation="sampling",
        runtime_seconds=elapsed,
        num_graphs=len(graphs),
        memory=memory_monitor.to_dict(),
    )
    attr_schema = normalize_schema(cfg)
    attr_postprocess_applied = False
    try:
        splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
        ref_graphs = list(splits.get("train", []))
    except Exception:
        ref_graphs = []
    attr_stats = fit_attribute_statistics(ref_graphs, attr_schema) if ref_graphs else fit_attribute_statistics([], attr_schema)
    attr_strategy = str(attr_schema.get("generated_attribute_strategy", "empirical")).lower()
    if attr_stats.has_any_attributes and attr_strategy == "empirical":
        before_cov = attribute_coverage(graphs, attr_schema)
        overwrite = bool(attr_schema.get("overwrite_generated_attributes", False))
        if overwrite or not before_cov.get("has_any_attributes", False):
            graphs = apply_empirical_attributes(graphs, attr_stats, seed=seed, overwrite=overwrite)
            assert_finite_graphs(graphs, context=f"attribute postprocessing {dataset}/{model_name} seed={seed}")
            attr_postprocess_applied = True
    quality = quality_metrics(graphs, reference_graphs=ref_graphs, dataset=dataset)
    trajectory_metadata = None
    if draw_trajectory:
        trajectory_path = trajectory_output or Path("outputs/figures/trajectories") / dataset / f"{model_name}_trajectory.png"
        trajectory_metadata = _draw_sampling_trajectory(
            dataset=dataset,
            model_name=model_name,
            reference_graphs=ref_graphs,
            generated_graphs=graphs,
            output_path=trajectory_path,
            num_graphs=trajectory_graphs,
            steps=trajectory_steps,
            seed=seed,
            node_label_attr=trajectory_node_label_attr,
            edge_label_attr=trajectory_edge_label_attr,
            show_node_labels=trajectory_show_node_labels,
            show_edge_labels=trajectory_show_edge_labels,
            dpi=trajectory_dpi,
        )
        logger.info("Saved sampling trajectory visualization to %s", trajectory_path)

    save_pickle(graphs, out, force=force)
    metadata = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "run_id": run_id,
        "num_samples_requested": num_samples,
        "num_samples_saved": len(graphs),
        "runtime_seconds": elapsed,
        "seconds_per_graph": elapsed / max(len(graphs), 1),
        "sampling_time_seconds": elapsed,
        "sampling_time_per_128_graphs_seconds": compute.get("seconds_per_128_graphs"),
        "hardware": compute["hardware_label"],
        "peak_memory_mib": compute.get("peak_memory_mib"),
        "compute": compute,
        "compute_budget": {
            "dataset": dataset,
            "model": model_name,
            "hardware": compute["hardware_label"],
            "training_time_seconds": None,
            "sampling_time_per_128_graphs_seconds": compute.get("seconds_per_128_graphs"),
            "peak_memory_mib": compute.get("peak_memory_mib"),
            "notes": f"Sampling measured over {len(graphs)} generated graphs and normalized to 128 graphs.",
        },
        "sample_path": str(out),
        "checkpoint_path": cfg.get("checkpoint_path"),
        "model_config_path": str(cfg_path),
        "model_config_hash": stable_hash(cfg),
        "capabilities": model_capabilities(model_name),
        "quality": quality,
        "trajectory_visualization": trajectory_metadata,
        "graph_attributes": {
            "schema": attr_schema,
            "fallback_attribute_postprocessing_applied": attr_postprocess_applied,
            "fallback_note": "If true, attributes were attached from empirical training-set marginals by the benchmark, not generated natively by the upstream model.",
            "train_attribute_stats": attr_stats.to_dict(),
            "train_attribute_coverage": attribute_coverage(ref_graphs, attr_schema) if ref_graphs else {},
            "generated_attribute_coverage": attribute_coverage(graphs, attr_schema),
        },
    }
    save_json(metadata, metadata_out, force=True)
    save_yaml(cfg, resolved_cfg_out, force=True)
    logger.info("Saved %d generated graphs to %s in %.2fs", len(graphs), out, elapsed)
    logger.info("Saved sample metadata to %s", metadata_out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate samples from a trained graph generator wrapper.")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed-offset", type=int, default=17, help="Offset added to the training seed for sampling.")
    parser.add_argument("--run-id", type=int, default=None, help="Optional repeated-run id. Enables run-specific checkpoint/sample paths when supplied.")
    parser.add_argument("--use-run-paths", action="store_true", help="Use run-specific checkpoint/sample paths even for run 0.")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the sampling progress bar.")
    parser.add_argument("--draw-trajectory", action="store_true", help="Draw reference-to-sample trajectory visualizations after sampling.")
    parser.add_argument("--trajectory-graphs", type=int, default=1, help="Number of generated samples to visualize as trajectory rows.")
    parser.add_argument("--trajectory-steps", type=int, default=8, help="Number of states per trajectory row, including reference and final sample.")
    parser.add_argument("--trajectory-output", type=str, default=None)
    parser.add_argument("--trajectory-node-label-attr", type=str, default="node_label")
    parser.add_argument("--trajectory-edge-label-attr", type=str, default="edge_type")
    parser.add_argument("--trajectory-show-node-labels", action="store_true")
    parser.add_argument("--trajectory-show-edge-labels", action="store_true")
    parser.add_argument("--trajectory-dpi", type=int, default=180)
    args = parser.parse_args()

    cfg_path = Path(args.model_config) if args.model_config else Path("configs/models") / f"{args.model}.yaml"
    base_cfg = load_yaml(cfg_path)
    _hide_cuda_for_cpu_config(base_cfg)
    _generate_samples(
        model_name=args.model,
        dataset=args.dataset,
        base_cfg=base_cfg,
        cfg_path=cfg_path,
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_seed_offset=args.sample_seed_offset,
        force=args.force,
        dry_run=args.dry_run,
        run_id=args.run_id,
        use_run_paths=bool(args.use_run_paths or args.run_id is not None),
        show_progress=not args.no_progress,
        draw_trajectory=args.draw_trajectory,
        trajectory_graphs=args.trajectory_graphs,
        trajectory_steps=args.trajectory_steps,
        trajectory_output=Path(args.trajectory_output) if args.trajectory_output else None,
        trajectory_node_label_attr=args.trajectory_node_label_attr,
        trajectory_edge_label_attr=args.trajectory_edge_label_attr,
        trajectory_show_node_labels=args.trajectory_show_node_labels,
        trajectory_show_edge_labels=args.trajectory_show_edge_labels,
        trajectory_dpi=args.trajectory_dpi,
    )


if __name__ == "__main__":
    main()
