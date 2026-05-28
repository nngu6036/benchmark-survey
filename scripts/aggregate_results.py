from __future__ import annotations

import argparse
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

from empirical_comparison.registry import available_datasets, available_models
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
    "run_id",
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
    if obj.get("run_id") is not None:
        row["run_id"] = obj.get("run_id")
    elif protocol.get("run_id") is not None:
        row["run_id"] = protocol.get("run_id")
    for k, v in (obj.get("results", {}) or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[k] = v
    return row


def _row_run_id(row: dict[str, Any]) -> int | None:
    value = row.get("run_id")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_by_run_ids(rows: list[dict[str, Any]], run_ids: set[int] | None) -> list[dict[str, Any]]:
    if run_ids is None:
        return rows
    filtered = []
    for row in rows:
        # Recompute requested subsets from individual run files. Existing
        # aggregate JSONs may cover a different set of run ids.
        if bool(row.get("is_aggregate", False)):
            continue
        row_run_id = _row_run_id(row)
        if row_run_id in run_ids or (row_run_id is None and 0 in run_ids):
            filtered.append(row)
    return filtered


def _filter_by_datasets_and_models(
    rows: list[dict[str, Any]],
    *,
    datasets: set[str] | None,
    models: set[str] | None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        row_dataset = str(row.get("dataset", "")).lower()
        row_model = str(row.get("model", "")).lower()
        if datasets is not None and row_dataset not in datasets:
            continue
        if models is not None and row_model not in models:
            continue
        filtered.append(row)
    return filtered


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


def _select_or_build_aggregate_rows(long_df: pd.DataFrame, *, prefer_existing_aggregates: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model", "metric_family"]
    for _, group in long_df.groupby(group_cols, dropna=False):
        aggregate_rows = group[group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else pd.DataFrame()
        if prefer_existing_aggregates and not aggregate_rows.empty:
            # Use the newest/last aggregate file for this metric family.
            rows.append(aggregate_rows.iloc[-1].to_dict())
        else:
            individual_rows = group[~group["is_aggregate"].fillna(False).astype(bool)] if "is_aggregate" in group.columns else group
            if not individual_rows.empty:
                rows.append(_aggregate_individual_metric_rows(individual_rows))
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
    parser = argparse.ArgumentParser(description="Aggregate metric JSON outputs into long, metric-family, and wide CSV tables.")
    parser.add_argument("--metric-dir", type=str, default="outputs/metrics")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=None, help="Datasets to include. Defaults to all discovered datasets.")
    parser.add_argument("--models", nargs="+", choices=available_models(), default=None, help="Models to include. Defaults to all discovered models.")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Only average these run ids. Existing aggregate JSONs are ignored when this is set.")
    args = parser.parse_args()

    metric_dir = Path(args.metric_dir)
    output_dir = Path(args.output_dir)
    requested_run_ids = set(args.run_ids) if args.run_ids is not None else None
    requested_datasets = {dataset.lower() for dataset in args.datasets} if args.datasets is not None else None
    requested_models = {model.lower() for model in args.models} if args.models is not None else None
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

    rows = _filter_by_datasets_and_models(rows, datasets=requested_datasets, models=requested_models)
    rows = _filter_by_run_ids(rows, requested_run_ids)

    if not rows:
        requested_parts = []
        if requested_datasets is not None:
            requested_parts.append(f"datasets={sorted(requested_datasets)}")
        if requested_models is not None:
            requested_parts.append(f"models={sorted(requested_models)}")
        if requested_run_ids is not None:
            requested_parts.append(f"run_ids={sorted(requested_run_ids)}")
        requested = f" matching {', '.join(requested_parts)}" if requested_parts else ""
        if requested_run_ids is None:
            logger.info("No metric files found under %s%s", metric_dir, requested)
        else:
            logger.info("No metric files found under %s%s", metric_dir, requested)
        return

    long_df = pd.DataFrame(rows)
    long_out = output_dir / "aggregated_results_long.csv"
    long_out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_out, index=False)

    selected_df = _select_or_build_aggregate_rows(long_df, prefer_existing_aggregates=requested_run_ids is None)
    selected_out = output_dir / "aggregated_results_by_metric_family.csv"
    selected_df.to_csv(selected_out, index=False)

    wide_df = _make_wide(selected_df)
    wide_out = output_dir / "aggregated_results.csv"
    wide_df.to_csv(wide_out, index=False)
    logger.info("Saved long results to %s", long_out)
    logger.info("Saved metric-family aggregate results to %s", selected_out)
    logger.info("Saved wide results to %s", wide_out)


if __name__ == "__main__":
    main()
