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


def _display_dataset(name: str) -> str:
    return "SBM" if str(name).lower() == "sbm" else str(name).capitalize()


def _display_model(name: str) -> str:
    return MODEL_NAMES.get(str(name).lower(), str(name).replace("_", r"\_"))


def _row(dataset: str, model: str) -> dict[str, str]:
    train_meta = _load_json(Path("outputs/runs") / dataset / model / "train_metadata.json")
    sample_meta = _load_json(Path("outputs/samples") / dataset / f"{model}.metadata.json")
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
        "training_time": _fmt_seconds(train_compute.get("training_time_seconds") or train_meta.get("training_time_seconds")),
        "sampling_time_per_128": _fmt_seconds(sample_compute.get("sampling_time_per_128_graphs_seconds") or sample_meta.get("sampling_time_per_128_graphs_seconds")),
        "peak_memory": _fmt_memory(max(peaks) if peaks else None),
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
    parser.add_argument("--datasets", nargs="*", choices=available_datasets(), default=["planar", "sbm"])
    parser.add_argument("--models", nargs="*", choices=available_models(), default=["graphguide", "digress", "construct", "edp_gnn", "disco", "grum"])
    parser.add_argument("--output", type=str, default="outputs/tables/compute_budget.tex")
    args = parser.parse_args()

    rows = [_row(dataset, model) for dataset in args.datasets for model in args.models]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_latex(rows) + "\n", encoding="utf-8")
    logger.info("Saved compute-budget table to %s", out)


if __name__ == "__main__":
    main()
