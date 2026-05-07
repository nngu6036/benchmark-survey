from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_run_ids(
    *,
    run_id: int | None = None,
    run_ids: Sequence[int] | None = None,
    num_runs: int = 1,
) -> list[int]:
    """Resolve CLI run arguments into an ordered list of integer run ids."""
    if run_id is not None and run_ids:
        raise ValueError("Use either --run-id or --run-ids, not both.")
    if run_id is not None:
        ids = [int(run_id)]
    elif run_ids:
        ids = [int(x) for x in run_ids]
    else:
        if num_runs <= 0:
            raise ValueError("num_runs must be positive.")
        ids = list(range(int(num_runs)))
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate run ids are not allowed: {ids}")
    if any(i < 0 for i in ids):
        raise ValueError(f"Run ids must be non-negative: {ids}")
    return ids


def should_use_run_paths(run_ids: Sequence[int], explicit_run_selection: bool = False) -> bool:
    """Use run-aware layout for multi-run experiments or explicit run selection."""
    return explicit_run_selection or len(run_ids) != 1 or (len(run_ids) == 1 and run_ids[0] != 0)


def run_seed(base_seed: int, run_id: int, seed_stride: int = 1000) -> int:
    return int(base_seed) + int(run_id) * int(seed_stride)


def _replace_placeholders(value: str, *, dataset: str, model: str, run_id: int | None, seed: int | None) -> str:
    replacements = {
        "dataset": dataset,
        "model": model,
        "run_id": "" if run_id is None else str(run_id),
        "run_id:03d": "" if run_id is None else f"{run_id:03d}",
        "seed": "" if seed is None else str(seed),
    }
    out = value
    # Support both {dataset} and ${dataset}.  We avoid str.format because YAML
    # configs may contain braces intended for other libraries.
    for key, repl in replacements.items():
        out = out.replace("${" + key + "}", repl)
        out = out.replace("{" + key + "}", repl)
    return out


def _contains_run_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(token in value for token in ("{run_id}", "${run_id}", "{run_id:03d}", "${run_id:03d}"))


def _run_dir_name(run_id: int) -> str:
    return f"run_{run_id:03d}"


def checkpoint_path_for_run(
    original: Any,
    *,
    dataset: str,
    model: str,
    run_id: int | None,
    seed: int,
    use_run_paths: bool,
) -> str | None:
    """Return the checkpoint path that a wrapper should use for this run.

    When run-aware paths are enabled, a config without an explicit run placeholder
    is redirected to outputs/checkpoints/{dataset}/{model}/run_xxx/<original name>.
    This prevents repeated training runs from overwriting one another while
    preserving file extensions expected by wrappers.
    """
    if original in (None, ""):
        if not use_run_paths or run_id is None:
            return None
        return str(Path("outputs/checkpoints") / dataset / model / _run_dir_name(run_id) / f"{model}.pt")

    if not isinstance(original, str):
        return original

    formatted = _replace_placeholders(original, dataset=dataset, model=model, run_id=run_id, seed=seed)
    if not use_run_paths or run_id is None or _contains_run_placeholder(original):
        return formatted

    original_name = Path(formatted).name or f"{model}.pt"
    return str(Path("outputs/checkpoints") / dataset / model / _run_dir_name(run_id) / original_name)


def directory_path_for_run(
    original: Any,
    *,
    dataset: str,
    model: str,
    run_id: int | None,
    seed: int,
    use_run_paths: bool,
    default_root: str,
) -> str | None:
    if original in (None, ""):
        if not use_run_paths or run_id is None:
            return None if original is None else ""
        return str(Path(default_root) / dataset / model / _run_dir_name(run_id))
    if not isinstance(original, str):
        return original
    formatted = _replace_placeholders(original, dataset=dataset, model=model, run_id=run_id, seed=seed)
    if not use_run_paths or run_id is None or _contains_run_placeholder(original):
        return formatted
    return str(Path(formatted) / _run_dir_name(run_id))


def make_model_run_config(
    model_cfg: dict[str, Any],
    *,
    dataset: str,
    model: str,
    run_id: int | None,
    seed: int,
    use_run_paths: bool,
) -> dict[str, Any]:
    """Deep-copy and resolve a model config for a specific training run."""
    cfg = copy.deepcopy(model_cfg)
    cfg["dataset"] = dataset
    cfg["model"] = model
    cfg["seed"] = int(seed)
    if run_id is not None:
        cfg["run_id"] = int(run_id)
        cfg["training_run_id"] = int(run_id)

    if "checkpoint_path" in cfg or use_run_paths:
        cfg["checkpoint_path"] = checkpoint_path_for_run(
            cfg.get("checkpoint_path"), dataset=dataset, model=model, run_id=run_id, seed=seed, use_run_paths=use_run_paths
        )

    # Common wrapper-owned directories.  They are redirected per run when needed
    # to keep provenance files and temporary upstream artifacts separate.
    directory_defaults = {
        "run_root": "outputs/model_runs",
        "data_root": "outputs/model_data",
        "data_subdir": "outputs/model_data",
        "data_dir": "outputs/model_data",
    }
    for key, default_root in directory_defaults.items():
        if key in cfg:
            cfg[key] = directory_path_for_run(
                cfg.get(key), dataset=dataset, model=model, run_id=run_id, seed=seed, use_run_paths=use_run_paths, default_root=default_root
            )

    # Resolve placeholders recursively for strings not handled above.
    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        if isinstance(obj, str):
            return _replace_placeholders(obj, dataset=dataset, model=model, run_id=run_id, seed=seed)
        return obj

    return resolve(cfg)


def run_output_dir(dataset: str, model: str, run_id: int | None = None) -> Path:
    base = Path("outputs/runs") / dataset / model
    return base if run_id is None else base / _run_dir_name(run_id)


def sample_path(dataset: str, model: str, run_id: int | None = None) -> Path:
    if run_id is None:
        return Path("outputs/samples") / dataset / f"{model}.pkl"
    return Path("outputs/samples") / dataset / model / f"{_run_dir_name(run_id)}.pkl"


def sample_metadata_path(dataset: str, model: str, run_id: int | None = None) -> Path:
    if run_id is None:
        return Path("outputs/samples") / dataset / f"{model}.metadata.json"
    return Path("outputs/samples") / dataset / model / f"{_run_dir_name(run_id)}.metadata.json"


def sample_config_path(dataset: str, model: str, run_id: int | None = None) -> Path:
    if run_id is None:
        return Path("outputs/samples") / dataset / f"{model}.resolved_model_config.yaml"
    return Path("outputs/samples") / dataset / model / f"{_run_dir_name(run_id)}.resolved_model_config.yaml"


def metric_path(dataset: str, model: str, metric_filename: str, run_id: int | None = None) -> Path:
    if run_id is None:
        return Path("outputs/metrics") / dataset / model / metric_filename
    return Path("outputs/metrics") / dataset / model / _run_dir_name(run_id) / metric_filename


def aggregate_metric_path(dataset: str, model: str, metric_filename: str) -> Path:
    stem = Path(metric_filename).stem
    return Path("outputs/metrics") / dataset / model / f"{stem}.aggregate.json"


def aggregate_numeric_results(run_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric `results` fields across repeated training runs."""
    import numpy as np

    values: dict[str, list[float]] = {}
    for payload in run_payloads:
        for key, value in (payload.get("results", {}) or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))

    summary: dict[str, Any] = {}
    nested: dict[str, dict[str, float | int]] = {}
    for key, vals in sorted(values.items()):
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        # Keep the bare key as the across-run mean so existing tables/plots still
        # work, and add explicit mean/std keys for repeated-run reporting.
        summary[key] = mean
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        nested[key] = {"mean": mean, "std": std, "num_runs": int(arr.size)}
    return {"flat": summary, "nested": nested}


def explicit_run_selection(run_id: int | None, run_ids: Sequence[int] | None) -> bool:
    return run_id is not None or bool(run_ids)


def existing_sample_path(dataset: str, model: str, run_id: int | None = None) -> Path:
    """Return the expected sample path, with legacy fallback for run 0."""
    path = sample_path(dataset, model, run_id)
    if path.exists():
        return path
    if run_id == 0:
        legacy = sample_path(dataset, model, None)
        if legacy.exists():
            return legacy
    return path
