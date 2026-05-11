from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from empirical_comparison.utils.logging import get_logger

logger = get_logger(__name__)

EXCLUDE_FROM_METRIC_AGG = {
    "dataset",
    "model",
    "metric_family",
    "source_file",
    "runtime_seconds",
    "seed",
    "base_seed",
    "is_aggregate",
}


def _flatten_results(obj: dict[str, Any]) -> dict[str, Any]:
    protocol = obj.get("protocol", {}) or {}
    row: dict[str, Any] = {
        "dataset": obj.get("dataset"),
        "model": obj.get("model"),
        "metric_family": obj.get("metric_family"),
        "runtime_seconds": obj.get("runtime_seconds"),
        "is_aggregate": bool(obj.get("is_aggregate", False)),
    }
    if "seed" in protocol:
        row["seed"] = protocol["seed"]
    if "base_seed" in protocol:
        row["base_seed"] = protocol["base_seed"]
    for k, v in (obj.get("results", {}) or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = v
    return row


def _is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series.dropna(), errors="coerce")
    return len(converted) > 0 and converted.notna().all()


def _aggregate_individual_metric_rows(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    out: dict[str, Any] = {
        "dataset": first.get("dataset"),
        "model": first.get("model"),
        "metric_family": first.get("metric_family"),
        "is_aggregate": True,
        "source_file": ";".join(group.get("source_file", pd.Series(dtype=str)).astype(str).tolist()),
    }
    if "runtime_seconds" in group.columns:
        vals = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
        if len(vals):
            out["runtime_seconds"] = float(vals.sum())
    for col in group.columns:
        if col in EXCLUDE_FROM_METRIC_AGG or col.endswith("_std"):
            continue
        values = group[col].dropna()
        if values.empty:
            continue
        if _is_numeric_series(values):
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            out[col] = mean
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
        else:
            unique = list(dict.fromkeys(map(str, values.tolist())))
            if len(unique) == 1:
                out[col] = unique[0]
    return out


def _select_or_build_aggregate_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model", "metric_family"]
    for _, group in long_df.groupby(group_cols, dropna=False):
        aggregate_rows = group[group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else pd.DataFrame()
        if not aggregate_rows.empty:
            # Use the newest/last aggregate file for this metric family.
            rows.append(aggregate_rows.iloc[-1].to_dict())
        else:
            rows.append(_aggregate_individual_metric_rows(group))
    return pd.DataFrame(rows)


def _make_wide(selected_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in selected_df.columns if c not in {"dataset", "model", "metric_family", "source_file"}]
    merged_rows = []
    for (dataset, model), group in selected_df.groupby(["dataset", "model"], dropna=False):
        out: dict[str, Any] = {"dataset": dataset, "model": model}
        if "runtime_seconds" in group:
            vals = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
            if len(vals):
                out["evaluation_runtime_seconds"] = float(vals.sum())
        for col in metric_cols:
            if col in {"runtime_seconds"}:
                continue
            if col not in group:
                continue
            vals = group[col].dropna().tolist()
            if len(vals) == 0:
                continue
            if len(vals) > 1 and len(set(map(str, vals))) > 1:
                logger.warning("Multiple different values for %s/%s/%s: %s; keeping first", dataset, model, col, vals)
            out[col] = vals[0]
        merged_rows.append(out)
    return pd.DataFrame(merged_rows).sort_values(["dataset", "model"])


def main() -> None:
    metric_dir = Path("outputs/metrics")
    rows = []
    for path in sorted(metric_dir.glob("**/*.json")):
        # Only metric JSONs should live under outputs/metrics, but skip hidden or
        # editor temp files defensively.
        if path.name.startswith("."):
            continue
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "results" not in obj:
            continue
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

    selected_df = _select_or_build_aggregate_rows(long_df)
    selected_out = Path("outputs/tables/aggregated_results_by_metric_family.csv")
    selected_df.to_csv(selected_out, index=False)

    wide_df = _make_wide(selected_df)
    wide_out = Path("outputs/tables/aggregated_results.csv")
    wide_df.to_csv(wide_out, index=False)
    logger.info("Saved long results to %s", long_out)
    logger.info("Saved metric-family aggregate results to %s", selected_out)
    logger.info("Saved wide results to %s", wide_out)


if __name__ == "__main__":
    main()
