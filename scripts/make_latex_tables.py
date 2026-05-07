from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


PREFERRED_COLUMNS = [
    "dataset",
    "model",
    "num_runs",
    "degree_mmd",
    "clustering_mmd",
    "orbit_mmd",
    "structural_summary_mmd",
    "spectral_mmd",
    "learned_feature_mmd",
    "polygraphscore",
    "pgs_js_distance",
    "classifier_auc",
    "dataset_validity_rate",
    "uniqueness_rate",
    "novelty_rate",
    "evaluation_runtime_seconds",
]

COLUMN_RENAMES = {
    "dataset": "Dataset",
    "model": "Model",
    "num_runs": "Runs",
    "degree_mmd": r"Degree MMD $\downarrow$",
    "clustering_mmd": r"Clustering MMD $\downarrow$",
    "orbit_mmd": r"Orbit MMD $\downarrow$",
    "structural_summary_mmd": r"Structural MMD $\downarrow$",
    "spectral_mmd": r"Spectral MMD $\downarrow$",
    "learned_feature_mmd": r"Feature MMD $\downarrow$",
    "polygraphscore": r"PGS $\downarrow$",
    "pgs_js_distance": r"PGS-JS $\downarrow$",
    "classifier_auc": r"Classifier AUC $\approx 0.5$",
    "dataset_validity_rate": r"Validity $\uparrow$",
    "uniqueness_rate": r"Unique $\uparrow$",
    "novelty_rate": r"Novel $\uparrow$",
    "evaluation_runtime_seconds": r"Eval. sec. $\downarrow$",
}


def _format_float(x: float) -> str:
    return f"{float(x):.4f}"


def _format_value(x):
    if pd.isna(x):
        return "--"
    if isinstance(x, float):
        return _format_float(x)
    return x


def _format_mean_std(row: pd.Series, col: str):
    x = row.get(col)
    if pd.isna(x):
        return "--"
    std_col = f"{col}_std"
    std = row.get(std_col) if std_col in row.index else None
    if std is not None and not pd.isna(std):
        try:
            return f"{float(x):.4f} $\\pm$ {float(std):.4f}"
        except Exception:
            pass
    return _format_value(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LaTeX table from aggregated benchmark CSV.")
    parser.add_argument("--input", type=str, default="outputs/tables/aggregated_results.csv")
    parser.add_argument("--output", type=str, default="outputs/tables/aggregated_results.tex")
    parser.add_argument("--mean-std", action="store_true", default=True, help="Render metric cells as mean ± std when *_std columns are available.")
    args = parser.parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.info("Aggregated results CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    cols = [c for c in PREFERRED_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No preferred metric columns found in aggregated results.")

    table_rows = []
    for _, row in df.iterrows():
        out = {}
        for col in cols:
            out[COLUMN_RENAMES.get(col, col)] = _format_mean_std(row, col) if args.mean_std else _format_value(row[col])
        table_rows.append(out)
    table_df = pd.DataFrame(table_rows)
    align = "ll" + "c" * max(0, len(table_df.columns) - 2)
    latex = table_df.to_latex(
        index=False,
        escape=False,
        caption=(
            "Aggregated graph-generation benchmark results. Repeated-training columns are reported as mean $\\pm$ standard deviation when available. "
            "Lower is better for discrepancy, PGS, and runtime metrics; classifier AUC is best near 0.5; validity, uniqueness, and novelty are higher-is-better."
        ),
        label="tab:synthetic_benchmark_results",
        column_format=align,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", out)


if __name__ == "__main__":
    main()
