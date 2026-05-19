from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import available_datasets
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


MOLECULAR_DATASETS = {"qm9"}
DEFAULT_MODELS = ["digress", "construct", "disco", "grum"]
MODEL_NAMES = {
    "construct": "ConStruct",
    "digress": "DiGress",
    "disco": "DisCo",
    "grum": "GruM",
}

METRIC_COLUMNS = [
    ("dataset_validity_rate", r"\makecell{Validity\\$\uparrow$}"),
    ("uniqueness_rate", r"\makecell{Uniqueness\\$\uparrow$}"),
    ("novelty_rate", r"\makecell{Novelty\\$\uparrow$}"),
    ("atom_type_mmd", r"\makecell{Atom-type\\MMD $\downarrow$}"),
    ("bond_type_mmd", r"\makecell{Bond-type\\MMD $\downarrow$}"),
    ("learned_feature_mmd", r"\makecell{Feature-\\space\\MMD $\downarrow$}"),
    ("pgs_js_distance", r"\makecell{PGS-JS\\$\downarrow$}"),
]


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _display_model(model: str) -> str:
    return MODEL_NAMES.get(model.lower(), model.replace("_", r"\_"))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(row: dict[str, str] | None, metric: str, *, mean_std: bool) -> str:
    if row is None:
        return "--"
    value = _as_float(row.get(metric))
    if value is None:
        return "--"
    if mean_std:
        for std_key in (f"{metric}_std", f"{metric}_bootstrap_std", f"{metric}_split_std"):
            std = _as_float(row.get(std_key))
            if std is not None:
                return f"{value:.4f} $\\pm$ {std:.4f}"
    return f"{value:.4f}"


def _find_rows(rows: list[dict[str, str]], dataset: str) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        row_dataset = str(row.get("dataset", "")).lower()
        row_model = str(row.get("model", "")).lower()
        if row_dataset == dataset.lower() and row_model:
            selected[row_model] = row
    return selected


def _latex(dataset_rows: dict[str, dict[str, str]], models: list[str], *, mean_std: bool) -> str:
    body = []
    for model in models:
        row = dataset_rows.get(model.lower())
        values = [_format_metric(row, metric, mean_std=mean_std) for metric, _ in METRIC_COLUMNS]
        body.append(f"{_display_model(model):<10} & " + " & ".join(values) + r" \\")

    header = ["Model", *(title for _, title in METRIC_COLUMNS)]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            (
                r"\caption{Illustrative reporting template for the QM9 molecular benchmark. Higher is better for validity, "
                r"uniqueness, and novelty; lower is better for MMD, feature-space MMD, and PGS-JS. GraphGUIDE and "
                r"EDP-GNN are omitted because their current benchmark implementations do not support attributed molecular graphs.}"
            ),
            r"\label{tab:qm9_benchmark_results}",
            r"\small",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabularx}{\textwidth}{l *{7}{>{\centering\arraybackslash}X}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LaTeX table for molecular benchmark results.")
    parser.add_argument("--dataset", choices=available_datasets(), default="qm9")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--input", type=str, default="outputs/tables/aggregated_results.csv")
    parser.add_argument("--output", type=str, default="outputs/tables/qm9_benchmark_results.tex")
    parser.add_argument("--mean-std", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if dataset not in MOLECULAR_DATASETS:
        raise ValueError(f"This table layout is for molecular datasets only; supported datasets: {sorted(MOLECULAR_DATASETS)}")

    rows = _load_rows(Path(args.input))
    dataset_rows = _find_rows(rows, dataset)
    missing = [model for model in args.models if model.lower() not in dataset_rows]
    if missing:
        logger.info("No aggregated rows found for %s/%s; cells will be rendered as --", dataset, ", ".join(missing))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_latex(dataset_rows, args.models, mean_std=args.mean_std) + "\n", encoding="utf-8")
    logger.info("Saved molecular benchmark table to %s", out)


if __name__ == "__main__":
    main()
