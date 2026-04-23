from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.reporting.latex import dataframe_to_latex
from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def main() -> None:
    csv_path = Path("outputs/tables/aggregated_results.csv")
    if not csv_path.exists():
        logger.info("Aggregated results CSV not found: %s", csv_path)
        return
    df = pd.read_csv(csv_path)
    latex = dataframe_to_latex(df, caption="Aggregated empirical comparison results.", label="tab:aggregated_results")
    out = Path("outputs/tables/aggregated_results.tex")
    out.write_text(latex, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", out)

if __name__ == "__main__":
    main()
