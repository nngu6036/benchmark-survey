from __future__ import annotations

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
    "degree_mmd",
    "clustering_mmd",
    "structural_summary_mmd",
    "spectral_mmd",
    "learned_feature_mmd",
    "classifier_auc_mean",
    "pgs_js_normalized_mean",
]

COLUMN_RENAMES = {
    "dataset": "Dataset",
    "model": "Model",
    "degree_mmd": "Degree MMD $\\downarrow$",
    "clustering_mmd": "Clustering MMD $\\downarrow$",
    "structural_summary_mmd": "Structural MMD $\\downarrow$",
    "spectral_mmd": "Spectral MMD $\\downarrow$",
    "learned_feature_mmd": "Random-feature MMD $\\downarrow$",
    "classifier_auc_mean": "Classifier AUC $\\downarrow$",
    "pgs_js_normalized_mean": "PGS-style JS $\\downarrow$",
}


def _format_value(x):
    if pd.isna(x):
        return "--"
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def main() -> None:
    csv_path = Path("outputs/tables/aggregated_results.csv")
    if not csv_path.exists():
        logger.info("Aggregated results CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    cols = [c for c in PREFERRED_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No preferred metric columns found in aggregated results.")

    table_df = df[cols].copy()
    table_df = table_df.rename(columns=COLUMN_RENAMES)
    table_df = table_df.applymap(_format_value)

    latex = table_df.to_latex(
        index=False,
        escape=False,
        caption="Illustrative comparison on synthetic graph benchmarks. Lower is better for discrepancy metrics; classifier AUC is best near 0.5.",
        label="tab:synthetic_benchmark_results",
        column_format="ll" + "c" * (len(table_df.columns) - 2),
    )

    out = Path("outputs/tables/aggregated_results.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", out)


if __name__ == "__main__":
    main()
