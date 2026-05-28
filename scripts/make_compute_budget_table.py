from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.evaluation.run_utils import run_output_dir, sample_metadata_path
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


MODEL_NAMES = {
    "construct": "ConStruct",
    "digress": "DiGress",
    "disco": "DisCo",
    "edp_gnn": "EDP-GNN",
    "graphguide": "GraphGUIDE",
    "grum": "GruM",
    "dummy": "Dummy",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "--"
    try:
        seconds = float(value)
    except Exception:
        return "--"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60.0:.2f}m"
    return f"{seconds / 3600.0:.2f}h"


def _fmt_memory(value: Any) -> str:
    if value is None:
        return "--"
    try:
        mib = float(value)
    except Exception:
        return "--"
    if mib >= 1024:
        return f"{mib / 1024.0:.2f} GiB"
    return f"{mib:.1f} MiB"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _display_dataset(name: str) -> str:
    return "SBM" if str(name).lower() == "sbm" else str(name).capitalize()


def _display_model(name: str) -> str:
    return MODEL_NAMES.get(str(name).lower(), str(name).replace("_", r"\_"))


def _metadata_paths(dataset: str, model: str, run_id: int | None) -> tuple[Path, Path]:
    return (
        run_output_dir(dataset, model, run_id=run_id) / "train_metadata.json",
        sample_metadata_path(dataset, model, run_id=run_id),
    )


def _row_from_metadata(dataset: str, model: str, train_meta_path: Path, sample_meta_path: Path) -> dict[str, Any]:
    train_meta = _load_json(train_meta_path)
    sample_meta = _load_json(sample_meta_path)
    train_compute = train_meta.get("compute_budget", {}) if isinstance(train_meta.get("compute_budget"), dict) else {}
    sample_compute = sample_meta.get("compute_budget", {}) if isinstance(sample_meta.get("compute_budget"), dict) else {}
    hardware = train_compute.get("hardware") or sample_compute.get("hardware") or train_meta.get("hardware") or sample_meta.get("hardware") or "--"
    peak_values = [
        train_compute.get("peak_memory_mib"),
        sample_compute.get("peak_memory_mib"),
        train_meta.get("peak_memory_mib"),
        sample_meta.get("peak_memory_mib"),
    ]
    peaks = []
    for value in peak_values:
        try:
            if value is not None:
                peaks.append(float(value))
        except Exception:
            pass
    notes = []
    if not train_meta:
        notes.append("missing train metadata")
    if not sample_meta:
        notes.append("missing sample metadata")
    return {
        "dataset": _display_dataset(dataset),
        "model": _display_model(model),
        "hardware": str(hardware),
        "training_time_seconds": _numeric(train_compute.get("training_time_seconds") or train_meta.get("training_time_seconds")),
        "sampling_time_per_128_seconds": _numeric(sample_compute.get("sampling_time_per_128_graphs_seconds") or sample_meta.get("sampling_time_per_128_graphs_seconds")),
        "peak_memory_mib": max(peaks) if peaks else None,
        "notes": "; ".join(notes) if notes else "--",
        "train_metadata_path": str(train_meta_path),
        "sample_metadata_path": str(sample_meta_path),
    }


def _row(dataset: str, model: str, run_ids: list[int] | None = None, *, debug_sources: bool = False) -> dict[str, str]:
    if run_ids is None:
        train_path, sample_path = _metadata_paths(dataset, model, run_id=None)
        row = _row_from_metadata(dataset, model, train_path, sample_path)
        if debug_sources:
            logger.info("%s/%s train metadata: %s", dataset, model, train_path)
            logger.info("%s/%s sample metadata: %s", dataset, model, sample_path)
        return {
            "dataset": str(row["dataset"]),
            "model": str(row["model"]),
            "hardware": str(row["hardware"]),
            "training_time": _fmt_seconds(row["training_time_seconds"]),
            "sampling_time_per_128": _fmt_seconds(row["sampling_time_per_128_seconds"]),
            "peak_memory": _fmt_memory(row["peak_memory_mib"]),
            "notes": str(row["notes"]),
        }

    run_rows = []
    for run_id in run_ids:
        train_path, sample_path = _metadata_paths(dataset, model, run_id=run_id)
        if debug_sources:
            logger.info("%s/%s run_id=%s train metadata: %s", dataset, model, run_id, train_path)
            logger.info("%s/%s run_id=%s sample metadata: %s", dataset, model, run_id, sample_path)
        run_rows.append(_row_from_metadata(dataset, model, train_path, sample_path))

    training_times = [v for v in (_numeric(row["training_time_seconds"]) for row in run_rows) if v is not None]
    sampling_times = [v for v in (_numeric(row["sampling_time_per_128_seconds"]) for row in run_rows) if v is not None]
    peak_memories = [v for v in (_numeric(row["peak_memory_mib"]) for row in run_rows) if v is not None]
    hardware_values = [str(row["hardware"]) for row in run_rows if row.get("hardware") and row.get("hardware") != "--"]
    notes = []
    missing_train = sum("missing train metadata" in str(row["notes"]) for row in run_rows)
    missing_sample = sum("missing sample metadata" in str(row["notes"]) for row in run_rows)
    if missing_train:
        notes.append(f"missing train metadata for {missing_train}/{len(run_rows)} runs")
    if missing_sample:
        notes.append(f"missing sample metadata for {missing_sample}/{len(run_rows)} runs")
    notes.append(f"averaged over run ids {','.join(map(str, run_ids))}")

    return {
        "dataset": _display_dataset(dataset),
        "model": _display_model(model),
        "hardware": hardware_values[0] if hardware_values else "--",
        "training_time": _fmt_seconds(_mean(training_times)),
        "sampling_time_per_128": _fmt_seconds(_mean(sampling_times)),
        "peak_memory": _fmt_memory(max(peak_memories) if peak_memories else None),
        "notes": "; ".join(notes) if notes else "--",
    }


def _latex(rows: list[dict[str, str]]) -> str:
    body = [
        (
            f"{r['dataset']} & {r['model']} & {r['hardware']} & {r['training_time']} & "
            f"{r['sampling_time_per_128']} & {r['peak_memory']} & {r['notes']} \\\\"
        )
        for r in rows
    ]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compute and runtime reporting. Runtime should be measured under a fixed hardware and software environment.}",
        r"\label{tab:appendix_compute_budget}",
        r"\small",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"Dataset & Model & Hardware & Training time & Sampling time / 128 graphs & Peak memory & Notes \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LaTeX compute-budget table from train/sample metadata.")
    parser.add_argument("--dataset", choices=available_datasets(), default=None, help="Single dataset to include.")
    parser.add_argument("--model", choices=available_models(), default=None, help="Single model to include.")
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=None, help="Datasets to include. Defaults to planar and SBM.")
    parser.add_argument("--models", nargs="+", choices=available_models(), default=None, help="Models to include. Defaults to all benchmark models.")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Use run-specific train/sample metadata and average times across these run ids.")
    parser.add_argument("--debug-sources", action="store_true", help="Log the train/sample metadata JSON paths used for each row.")
    parser.add_argument("--output", type=str, default="outputs/tables/compute_budget.tex")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else (args.datasets if args.datasets is not None else ["planar", "sbm"])
    models = [args.model] if args.model else (args.models if args.models is not None else ["graphguide", "digress", "construct", "edp_gnn", "disco", "grum"])

    rows = [_row(dataset, model, args.run_ids, debug_sources=args.debug_sources) for dataset in datasets for model in models]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_latex(rows) + "\n", encoding="utf-8")
    logger.info("Saved compute-budget table to %s", out)


if __name__ == "__main__":
    main()
