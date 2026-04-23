from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def main() -> None:
    rows = []
    for path in Path("outputs/metrics").glob("*/*/*.json"):
        with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
        row = {"dataset": obj["dataset"], "model": obj["model"]}
        row.update(obj.get("results", {}))
        rows.append(row)
    if not rows:
        logger.info("No metric files found.")
        return
    df = pd.DataFrame(rows).groupby(["dataset", "model"], as_index=False).first()
    out = Path("outputs/tables/aggregated_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved aggregated results to %s", out)

if __name__ == "__main__":
    main()
