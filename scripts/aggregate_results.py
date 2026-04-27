from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)


def _flatten_results(obj: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": obj.get("dataset"),
        "model": obj.get("model"),
        "metric_family": obj.get("metric_family"),
    }
    for k, v in obj.get("results", {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = v
    return row


def main() -> None:
    metric_dir = Path("outputs/metrics")
    rows = []
    for path in sorted(metric_dir.glob("*/*/*.json")):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        row = _flatten_results(obj)
        row["source_file"] = str(path)
        rows.append(row)

    if not rows:
        logger.info("No metric files found under %s", metric_dir)
        return

    long_df = pd.DataFrame(rows)
    long_out = Path("outputs/tables/aggregated_results_long.csv")
    long_out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_out, index=False)

    # Merge rows per dataset/model without silently dropping non-null metric columns.
    metric_cols = [c for c in long_df.columns if c not in {"dataset", "model", "metric_family", "source_file"}]
    merged_rows = []
    for (dataset, model), group in long_df.groupby(["dataset", "model"], dropna=False):
        out = {"dataset": dataset, "model": model}
        for col in metric_cols:
            vals = group[col].dropna().tolist()
            if len(vals) == 0:
                continue
            if len(vals) > 1 and len(set(map(str, vals))) > 1:
                logger.warning("Multiple different values for %s/%s/%s: %s; keeping first", dataset, model, col, vals)
            out[col] = vals[0]
        merged_rows.append(out)

    wide_df = pd.DataFrame(merged_rows).sort_values(["dataset", "model"])
    wide_out = Path("outputs/tables/aggregated_results.csv")
    wide_df.to_csv(wide_out, index=False)
    logger.info("Saved long results to %s", long_out)
    logger.info("Saved wide results to %s", wide_out)


if __name__ == "__main__":
    main()
