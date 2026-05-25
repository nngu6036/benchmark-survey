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


MOLECULAR_DATASETS = {"qm9", "zinc"}
DEFAULT_MODELS = ["digress", "construct", "disco", "grum"]
DEFAULT_DATASETS = ["qm9", "zinc"]
MODEL_NAMES = {
    "construct": "ConStruct",
    "digress": "DiGress",
    "disco": "DisCo",
    "grum": "GruM",
}

METRIC_COLUMNS = [
    ("atom_type_mmd", r"\makecell{Atom-type\\MMD}"),
    ("bond_type_mmd", r"\makecell{Bond-type\\MMD}"),
    ("validity_rate", r"\makecell{Validity\\$\uparrow$}"),
    ("uniqueness_rate", r"\makecell{Uniqueness\\$\uparrow$}"),
    ("novelty_rate", r"\makecell{Novelty\\$\uparrow$}"),
    ("learned_feature_mmd", r"\makecell{Feature-\\space\\MMD}"),
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


def _find_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_dataset = str(row.get("dataset", "")).lower()
        row_model = str(row.get("model", "")).lower()
        if row_dataset and row_model:
            selected[(row_dataset, row_model)] = row
    return selected


def _latex(rows_by_key: dict[tuple[str, str], dict[str, str]], datasets: list[str], models: list[str], *, mean_std: bool) -> str:
    body = []
    for dataset_idx, dataset in enumerate(datasets):
        if dataset_idx > 0:
            body.append(r"\midrule")
        for model in models:
            row = rows_by_key.get((dataset.lower(), model.lower()))
            values = [_format_metric(row, metric, mean_std=mean_std) for metric, _ in METRIC_COLUMNS]
            body.append(f"{dataset.upper():<4} & {_display_model(model):<10} & " + " & ".join(values) + r" \\")

    header = ["Dataset", "Model", *(title for _, title in METRIC_COLUMNS)]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            (
                r"\caption{Illustrative reporting template for molecular benchmarks. Lower is better for atom-type MMD, "
                r"bond-type MMD, feature-space MMD, and PGS-JS; higher is better for validity, uniqueness, and novelty. "
                r"Entries are left blank until the molecular evaluation is completed. GraphGUIDE and EDP-GNN are omitted "
                r"because their current benchmark implementations do not support attributed molecular graphs.}"
            ),
            r"\label{tab:molecular_benchmark_results}",
            r"\small",
            r"\setlength{\tabcolsep}{2.8pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabularx}{\textwidth}{l l *{7}{>{\centering\arraybackslash}X}}",
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
    parser.add_argument("--dataset", choices=available_datasets(), default=None, help="Single molecular dataset to include. Prefer --datasets for the full table.")
    parser.add_argument("--datasets", nargs="*", choices=available_datasets(), default=None, help="Molecular datasets to include. Defaults to QM9 and ZINC.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--input", type=str, default="outputs/tables/aggregated_results.csv")
    parser.add_argument("--output", type=str, default="outputs/tables/molecular_benchmark_results.tex")
    parser.add_argument("--mean-std", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    datasets = [d.lower() for d in (args.datasets or ([args.dataset] if args.dataset else DEFAULT_DATASETS))]
    unsupported = [d for d in datasets if d not in MOLECULAR_DATASETS]
    if unsupported:
        raise ValueError(f"This table layout is for molecular datasets only; supported datasets: {sorted(MOLECULAR_DATASETS)}")

    rows = _load_rows(Path(args.input))
    rows_by_key = _find_rows(rows)
    missing = [f"{dataset}/{model}" for dataset in datasets for model in args.models if (dataset, model.lower()) not in rows_by_key]
    if missing:
        logger.info("No aggregated rows found for %s; cells will be rendered as --", ", ".join(missing))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_latex(rows_by_key, datasets, args.models, mean_std=args.mean_std) + "\n", encoding="utf-8")
    logger.info("Saved molecular benchmark table to %s", out)


if __name__ == "__main__":
    main()
