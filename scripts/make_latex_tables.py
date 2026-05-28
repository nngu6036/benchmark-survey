from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.registry import available_datasets, available_models
from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


SYNTHETIC_DATASETS = {"planar", "sbm"}
QM9_MODELS = ["digress", "construct", "disco", "grum"]
MODEL_NAMES = {
    "construct": "ConStruct",
    "digress": "DiGress",
    "disco": "DisCo",
    "edp_gnn": "EDP-GNN",
    "graphguide": "GraphGUIDE",
    "grum": "GruM",
    "dummy": "Dummy",
}

SYNTHETIC_COLUMNS = [
    ("degree_mmd", r"\makecell{Degree\\MMD}"),
    ("clustering_mmd", r"\makecell{Clustering\\MMD}"),
    ("orbit_mmd", r"\makecell{Orbit\\MMD}"),
    ("spectral_mmd", r"\makecell{Spectral\\MMD}"),
    ("learned_feature_mmd", r"\makecell{Feature-\\space\\MMD}"),
    ("pgs_js_distance", r"\makecell{PGS-JS\\$\downarrow$}"),
]

QM9_COLUMNS = [
    ("dataset_validity_rate", r"\makecell{Validity\\$\uparrow$}"),
    ("uniqueness_rate", r"\makecell{Uniqueness\\$\uparrow$}"),
    ("novelty_rate", r"\makecell{Novelty\\$\uparrow$}"),
    ("atom_type_mmd", r"\makecell{Atom-type\\MMD $\downarrow$}"),
    ("bond_type_mmd", r"\makecell{Bond-type\\MMD $\downarrow$}"),
    ("learned_feature_mmd", r"\makecell{Feature-\\space\\MMD $\downarrow$}"),
    ("pgs_js_distance", r"\makecell{PGS-JS\\$\downarrow$}"),
]


def _format_float(x: float) -> str:
    return f"{float(x):.4f}"


def _format_value(x):
    if pd.isna(x):
        return "--"
    if isinstance(x, float):
        return _format_float(x)
    return x


def _format_mean_std(row: pd.Series, col: str) -> str:
    x = row.get(col)
    if pd.isna(x):
        return "--"
    std = None
    for std_col in (f"{col}_std", f"{col}_bootstrap_std", f"{col}_split_std"):
        if std_col in row.index and not pd.isna(row.get(std_col)):
            std = row.get(std_col)
            break
    if std is not None:
        try:
            return f"{float(x):.4f} $\\pm$ {float(std):.4f}"
        except Exception:
            pass
    return str(_format_value(x))


def _numeric_value(row: pd.Series, col: str) -> float | None:
    try:
        value = row.get(col)
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _display_model(value: str) -> str:
    key = str(value)
    return MODEL_NAMES.get(key.lower(), key.replace("_", r"\_"))


def _display_dataset(value: str) -> str:
    key = str(value).lower()
    if key == "sbm":
        return "SBM"
    return str(value).capitalize()


def _ordered_rows(df: pd.DataFrame, *, qm9: bool = False) -> pd.DataFrame:
    out = df.copy()
    if qm9:
        order = {name: i for i, name in enumerate(QM9_MODELS)}
        out["_model_order"] = out["model"].astype(str).str.lower().map(order).fillna(999)
        return out.sort_values(["_model_order", "model"]).drop(columns=["_model_order"])

    dataset_order = {"planar": 0, "sbm": 1}
    model_order = {name: i for i, name in enumerate(["construct", "digress", "disco", "edp_gnn", "graphguide", "grum", "dummy"])}
    out["_dataset_order"] = out["dataset"].astype(str).str.lower().map(dataset_order).fillna(999)
    out["_model_order"] = out["model"].astype(str).str.lower().map(model_order).fillna(999)
    return out.sort_values(["_dataset_order", "_model_order", "dataset", "model"]).drop(columns=["_dataset_order", "_model_order"])


def _render_table(
    df: pd.DataFrame,
    *,
    metric_columns: list[tuple[str, str]],
    caption: str,
    label: str,
    include_dataset: bool,
    mean_std: bool,
    tabcolsep: str = "2.5pt",
    midrule_on_dataset_change: bool = False,
    bold_best: bool = False,
) -> str:
    if df.empty:
        return ""

    leading = "l l" if include_dataset else "l"
    colspec = f"{leading} *{{{len(metric_columns)}}}{{>{{\\centering\\arraybackslash}}X}}"
    header = ["Dataset", "Model"] if include_dataset else ["Model"]
    header.extend(title for _, title in metric_columns)

    best_by_dataset: dict[tuple[str, str], float] = {}
    if bold_best and include_dataset:
        for dataset, group in df.groupby(df["dataset"].astype(str).str.lower(), dropna=False):
            for col, _ in metric_columns:
                values = [_numeric_value(row, col) for _, row in group.iterrows()]
                values = [value for value in values if value is not None]
                if values:
                    best_by_dataset[(str(dataset), col)] = min(values)

    rows = []
    previous_dataset = None
    for _, row in df.iterrows():
        current_dataset = str(row["dataset"]).lower() if include_dataset else None
        if midrule_on_dataset_change and previous_dataset is not None and current_dataset != previous_dataset:
            rows.append(r"\midrule")
        previous_dataset = current_dataset
        fields = []
        if include_dataset:
            fields.append(_display_dataset(row["dataset"]))
        fields.append(_display_model(row["model"]))
        for col, _ in metric_columns:
            value = _format_mean_std(row, col) if mean_std else str(_format_value(row.get(col)))
            numeric = _numeric_value(row, col)
            best = best_by_dataset.get((str(current_dataset), col)) if current_dataset is not None else None
            if bold_best and numeric is not None and best is not None and abs(numeric - best) <= 1e-12:
                value = rf"\textbf{{{value}}}"
            fields.append(value)
        rows.append(" & ".join(fields) + r" \\")

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\small",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            r"\renewcommand{\arraystretch}{1.12}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{colspec}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
        ]
    )


def _build_latex_tables(synthetic_df: pd.DataFrame, qm9_df: pd.DataFrame, *, mean_std: bool, label_suffix: str = "") -> str:
    tables = [
        _render_table(
            synthetic_df,
            metric_columns=SYNTHETIC_COLUMNS,
            caption="Illustrative comparison on synthetic graph benchmarks. Lower is better for MMD, feature-space MMD, and PGS-JS.",
            label=f"tab:synthetic_benchmark_results{label_suffix}",
            include_dataset=True,
            mean_std=mean_std,
            tabcolsep="3.0pt",
            midrule_on_dataset_change=True,
            bold_best=True,
        ),
        _render_table(
            qm9_df,
            metric_columns=QM9_COLUMNS,
            caption=(
                "Illustrative comparison on molecular graph benchmarks. Higher is better for validity, uniqueness, and novelty; "
                "lower is better for MMD, feature-space MMD, and PGS-JS."
            ),
            label=f"tab:qm9_benchmark_results{label_suffix}",
            include_dataset=True,
            mean_std=mean_std,
        ),
    ]
    return "\n\n".join(table for table in tables if table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LaTeX benchmark tables from aggregated benchmark CSV.")
    parser.add_argument("--input", type=str, default="outputs/tables/aggregated_results.csv")
    parser.add_argument("--output", type=str, default="outputs/tables/aggregated_results.tex")
    parser.add_argument("--simple-output", type=str, default=None, help="Path for the simplified table without standard deviations. Defaults to <output stem>_simple.tex.")
    parser.add_argument("--dataset", choices=available_datasets(), default=None, help="Single dataset to include.")
    parser.add_argument("--model", choices=available_models(), default=None, help="Single model to include.")
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=None, help="Datasets to include. Defaults to all supported rows in the input CSV.")
    parser.add_argument("--models", nargs="+", choices=available_models(), default=None, help="Models to include. Defaults to all models in the input CSV.")
    parser.add_argument("--mean-std", action=argparse.BooleanOptionalAction, default=True, help="Deprecated; the full table always includes standard deviations when available.")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.info("Aggregated results CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    if not {"dataset", "model"}.issubset(df.columns):
        raise ValueError("Aggregated results must include dataset and model columns.")

    df["dataset_key"] = df["dataset"].astype(str).str.lower()
    df["model_key"] = df["model"].astype(str).str.lower()
    datasets = [args.dataset] if args.dataset else args.datasets
    models = [args.model] if args.model else args.models
    if datasets is not None:
        dataset_filter = {dataset.lower() for dataset in datasets}
        df = df[df["dataset_key"].isin(dataset_filter)]
    if models is not None:
        model_filter = {model.lower() for model in models}
        df = df[df["model_key"].isin(model_filter)]

    synthetic_df = _ordered_rows(df[df["dataset_key"].isin(SYNTHETIC_DATASETS)].drop(columns=["dataset_key"]))
    qm9_df = _ordered_rows(df[df["dataset_key"].eq("qm9")].drop(columns=["dataset_key"]), qm9=True)

    full_latex = _build_latex_tables(synthetic_df, qm9_df, mean_std=True)
    simple_latex = _build_latex_tables(synthetic_df, qm9_df, mean_std=False, label_suffix="_simple")
    if not full_latex and not simple_latex:
        raise ValueError("No synthetic or QM9 rows found in aggregated results.")

    preamble = (
        "% Requires \\usepackage{booktabs,tabularx,makecell,array}\n"
        "% Generated from evaluated metrics: descriptor MMDs, feature-space MMD, PGS-style JS, and QM9 attribute/validity metrics.\n\n"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(preamble + full_latex + "\n", encoding="utf-8")

    simple_out = Path(args.simple_output) if args.simple_output else out.with_name(f"{out.stem}_simple{out.suffix}")
    simple_out.parent.mkdir(parents=True, exist_ok=True)
    simple_out.write_text(preamble + simple_latex + "\n", encoding="utf-8")
    logger.info("Saved full LaTeX table to %s", out)
    logger.info("Saved simplified LaTeX table to %s", simple_out)


if __name__ == "__main__":
    main()
