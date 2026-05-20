from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.utils.io import load_yaml, save_json


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def estimate(
    *,
    model_cfg: dict,
    dataset_cfg: dict,
    num_generated_graphs: int,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    sample_batch_size: int | None = None,
) -> dict:
    # These defaults mirror external/DiGress/configs/model/discrete.yaml.
    model_overrides = model_cfg.get("model_overrides") or {}
    hidden = model_overrides.get("hidden_dims") or {
        "dx": 256,
        "de": 64,
        "dy": 64,
        "n_head": 8,
        "dim_ffX": 256,
        "dim_ffE": 128,
        "dim_ffy": 128,
    }
    hidden_mlp = model_overrides.get("hidden_mlp_dims") or {"X": 256, "E": 128, "y": 128}
    L = int(model_overrides.get("n_layers", 5))
    T = int(model_overrides.get("diffusion_steps", _get(model_cfg, "sampling.num_steps", 500)))
    B = int(batch_size or model_cfg.get("batch_size", 32))
    Bs = int(sample_batch_size or model_cfg.get("sample_batch_size", B))
    n = int(dataset_cfg.get("num_nodes", 64))
    epochs = int(num_epochs or model_cfg.get("num_epochs", 100))
    train_fraction = float(_get(dataset_cfg, "split.train", 0.8))
    num_train = int(round(int(dataset_cfg.get("num_graphs", 10240)) * train_fraction))
    train_steps_per_epoch = math.ceil(num_train / max(1, B))
    train_steps = train_steps_per_epoch * epochs
    sample_batches = math.ceil(num_generated_graphs / max(1, Bs))

    dx, de, dy = int(hidden["dx"]), int(hidden["de"]), int(hidden["dy"])
    dim_ffX, dim_ffE, dim_ffy = int(hidden["dim_ffX"]), int(hidden["dim_ffE"]), int(hidden.get("dim_ffy", hidden["dim_ffX"]))
    hX, hE, hy = int(hidden_mlp["X"]), int(hidden_mlp["E"]), int(hidden_mlp["y"])
    # Synthetic SPECTRE defaults with extra_features='all': X_t + cycles/eigvecs, E_t only, y_t + t + graph features.
    input_x_dim = 7
    input_e_dim = 2
    input_y_dim = 14
    out_x_dim = 1
    out_e_dim = 2
    out_y_dim = 0

    input_macs = B * n * (input_x_dim * hX + hX * dx) + B * n * n * (input_e_dim * hE + hE * de) + B * (input_y_dim * hy + hy * dy)
    layer_macs = (
        3 * B * n * dx * dx
        + 3 * B * n * n * dx
        + 3 * B * n * n * de * dx
        + B * n * dx * dx
        + B * n * (dx * dim_ffX + dim_ffX * dx)
        + B * n * n * (de * dim_ffE + dim_ffE * de)
        + B * (4 * dx * dy + 4 * de * dy + 4 * dy * dx + 3 * dy * dy + dy * dim_ffy + dim_ffy * dy)
    )
    output_macs = B * n * (dx * hX + hX * out_x_dim) + B * n * n * (de * hE + hE * out_e_dim) + B * (dy * hy + hy * out_y_dim)
    forward_macs = int(input_macs + L * layer_macs + output_macs)
    forward_flops = 2 * forward_macs
    train_flops = 3 * forward_flops * train_steps  # rough fwd+bwd+optimizer multiplier

    # Sample complexity scales with sample batch size. Recompute forward MACs if sample_batch_size differs.
    if Bs != B:
        sample_model_cfg = dict(model_cfg)
        sample_est = estimate(
            model_cfg={**sample_model_cfg, "batch_size": Bs, "sample_batch_size": Bs},
            dataset_cfg=dataset_cfg,
            num_generated_graphs=num_generated_graphs,
            num_epochs=epochs,
            batch_size=Bs,
            sample_batch_size=Bs,
        )
        sample_forward_flops = sample_est["forward_flops_per_batch"]
    else:
        sample_forward_flops = forward_flops
    sampling_forward_calls = sample_batches * T
    sampling_flops = sample_forward_flops * sampling_forward_calls

    return {
        "num_nodes": n,
        "train_graphs": num_train,
        "batch_size": B,
        "sample_batch_size": Bs,
        "num_epochs": epochs,
        "diffusion_steps": T,
        "layers": L,
        "hidden_dims": hidden,
        "train_steps_per_epoch": train_steps_per_epoch,
        "train_steps": train_steps,
        "sample_batches": sample_batches,
        "sampling_forward_calls": sampling_forward_calls,
        "forward_macs_per_batch": forward_macs,
        "forward_flops_per_batch": forward_flops,
        "training_flops_approx": int(train_flops),
        "sampling_flops_approx": int(sampling_flops),
    }


def add_runtime_ranges(est: dict, *, train_eff_tflops: tuple[float, float], sample_eff_tflops: tuple[float, float], train_overhead: float, sample_overhead: float) -> dict:
    low_train, high_train = train_eff_tflops
    low_sample, high_sample = sample_eff_tflops
    train_seconds_min = est["training_flops_approx"] / (high_train * 1e12) * train_overhead
    train_seconds_max = est["training_flops_approx"] / (low_train * 1e12) * train_overhead
    sample_seconds_min = est["sampling_flops_approx"] / (high_sample * 1e12) * sample_overhead
    sample_seconds_max = est["sampling_flops_approx"] / (low_sample * 1e12) * sample_overhead
    return {
        **est,
        "runtime_model": {
            "train_effective_tflops_range": [low_train, high_train],
            "sample_effective_tflops_range": [low_sample, high_sample],
            "train_overhead_factor": train_overhead,
            "sample_overhead_factor": sample_overhead,
            "train_seconds_range": [train_seconds_min, train_seconds_max],
            "train_hours_range": [train_seconds_min / 3600, train_seconds_max / 3600],
            "sample_seconds_range": [sample_seconds_min, sample_seconds_max],
            "sample_minutes_range": [sample_seconds_min / 60, sample_seconds_max / 60],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate DiGress dense-operation counts and rough A6000 runtime ranges.")
    parser.add_argument("--dataset", choices=["sbm", "planar"], default="sbm")
    parser.add_argument("--model-config", default="configs/models/digress.yaml")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--num-generated-graphs", type=int, default=1024)
    parser.add_argument("--effective-train-tflops", nargs=2, type=float, default=[1.0, 3.0])
    parser.add_argument("--effective-sample-tflops", nargs=2, type=float, default=[0.6, 2.0])
    parser.add_argument("--train-overhead-factor", type=float, default=2.0)
    parser.add_argument("--sample-overhead-factor", type=float, default=2.5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    dataset_cfg_path = args.dataset_config or f"configs/datasets/{args.dataset}.yaml"
    dataset_cfg = load_yaml(dataset_cfg_path)
    est = estimate(model_cfg=model_cfg, dataset_cfg=dataset_cfg, num_generated_graphs=args.num_generated_graphs)
    est = add_runtime_ranges(
        est,
        train_eff_tflops=(args.effective_train_tflops[0], args.effective_train_tflops[1]),
        sample_eff_tflops=(args.effective_sample_tflops[0], args.effective_sample_tflops[1]),
        train_overhead=args.train_overhead_factor,
        sample_overhead=args.sample_overhead_factor,
    )
    print(f"DiGress estimate for {args.dataset}")
    print(f"  train graphs: {est['train_graphs']}, nodes: {est['num_nodes']}, epochs: {est['num_epochs']}")
    print(f"  train steps: {est['train_steps']} ({est['train_steps_per_epoch']} per epoch)")
    print(f"  forward FLOPs/batch: {est['forward_flops_per_batch'] / 1e9:.1f} GFLOP")
    print(f"  training FLOPs approx: {est['training_flops_approx'] / 1e15:.2f} PFLOP")
    print(f"  sampling forward calls: {est['sampling_forward_calls']} ({est['sample_batches']} batches × {est['diffusion_steps']} steps)")
    print(f"  sampling FLOPs approx: {est['sampling_flops_approx'] / 1e15:.2f} PFLOP")
    rt = est["runtime_model"]
    print(f"  estimated training time: {rt['train_hours_range'][0]:.2f}–{rt['train_hours_range'][1]:.2f} h")
    print(f"  estimated 1024-sample time: {rt['sample_minutes_range'][0]:.1f}–{rt['sample_minutes_range'][1]:.1f} min")
    if args.output_json:
        save_json(est, args.output_json, force=True)


if __name__ == "__main__":
    main()
