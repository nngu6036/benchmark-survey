from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.evaluation.run_utils import run_output_dir, sample_metadata_path, sample_path
from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATASETS = ["planar", "sbm", "qm9", "zinc"]
DEFAULT_MODELS = ["digress", "disco", "edp_gnn", "construct", "graphguide", "grum"]
RUN_RE = re.compile(r"run_(\d+)")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sample_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        return len(payload) if hasattr(payload, "__len__") else None
    except Exception:
        return None


def _run_id_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = RUN_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    match = RUN_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _discover_run_ids(dataset: str, model: str) -> list[int | None]:
    run_ids: set[int | None] = set()
    base_run_dir = run_output_dir(dataset, model)
    if base_run_dir.exists():
        if (base_run_dir / "train_metadata.json").exists():
            run_ids.add(None)
        for path in base_run_dir.glob("run_*/train_metadata.json"):
            run_ids.add(_run_id_from_path(path))

    legacy_sample_meta = sample_metadata_path(dataset, model, run_id=None)
    if legacy_sample_meta.exists() or sample_path(dataset, model, run_id=None).exists():
        run_ids.add(None)
    sample_dir = Path("outputs/samples") / dataset / model
    if sample_dir.exists():
        for path in sample_dir.glob("run_*.metadata.json"):
            run_ids.add(_run_id_from_path(path))
        for path in sample_dir.glob("run_*.pkl"):
            run_ids.add(_run_id_from_path(path))
    return sorted(run_ids, key=lambda item: (-1 if item is None else int(item)))


def _status_for(dataset: str, model: str, run_id: int | None) -> dict[str, Any]:
    train_meta_path = run_output_dir(dataset, model, run_id=run_id) / "train_metadata.json"
    train_meta = _load_json(train_meta_path)
    checkpoint = Path(str(train_meta.get("checkpoint_path") or "")) if train_meta.get("checkpoint_path") else None
    sample_meta_path = sample_metadata_path(dataset, model, run_id=run_id)
    sample_meta = _load_json(sample_meta_path)
    samples = sample_path(dataset, model, run_id=run_id)
    sample_count = _sample_count(samples)
    return {
        "dataset": dataset,
        "model": model,
        "run_id": "" if run_id is None else int(run_id),
        "training_complete": bool(train_meta_path.exists() and checkpoint is not None and checkpoint.exists()),
        "checkpoint_path": str(checkpoint) if checkpoint is not None else "",
        "training_metadata": str(train_meta_path) if train_meta_path.exists() else "",
        "training_seconds": train_meta.get("training_time_seconds", train_meta.get("runtime_seconds", "")),
        "sampling_complete": bool(samples.exists()),
        "num_samples": sample_count if sample_count is not None else sample_meta.get("num_samples_saved", ""),
        "sample_path": str(samples) if samples.exists() else "",
        "sampling_metadata": str(sample_meta_path) if sample_meta_path.exists() else "",
        "sampling_seconds": sample_meta.get("sampling_time_seconds", sample_meta.get("runtime_seconds", "")),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "model",
        "run_id",
        "training_complete",
        "num_samples",
        "sampling_complete",
        "training_metadata",
        "sampling_metadata",
    ]
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report training and sampling status for benchmark runs.")
    parser.add_argument("--datasets", nargs="*", choices=available_datasets(), default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="*", choices=available_models(), default=DEFAULT_MODELS)
    parser.add_argument("--run-ids", type=int, nargs="*", default=None, help="Explicit run ids to check. By default, discover run ids from metadata/sample files.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            run_ids = [int(x) for x in args.run_ids] if args.run_ids is not None else _discover_run_ids(dataset, model)
            if not run_ids:
                rows.append(_status_for(dataset, model, run_id=None))
                continue
            for run_id in run_ids:
                rows.append(_status_for(dataset, model, run_id=run_id))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved run status report to %s", out)
    _print_table(rows)


if __name__ == "__main__":
    main()
