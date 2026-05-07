from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def bar_plot_metric(df: pd.DataFrame, metric: str, output_path: str | Path) -> Path:
    """Create a simple metric bar plot grouped by dataset/model."""
    if metric not in df.columns:
        raise KeyError(f"Metric column not found: {metric}")
    plot_df = df[["dataset", "model", metric]].dropna().copy()
    plot_df["label"] = plot_df["dataset"].astype(str) + " / " + plot_df["model"].astype(str)
    ax = plot_df.plot(kind="bar", x="label", y=metric, legend=False)
    ax.set_xlabel("Dataset / model")
    ax.set_ylabel(metric)
    ax.figure.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, bbox_inches="tight")
    plt.close(ax.figure)
    return out


def make_default_metric_plots(
    csv_path: str | Path = "outputs/tables/aggregated_results.csv",
    output_dir: str | Path = "outputs/figures",
    metrics: Iterable[str] = ("degree_mmd", "clustering_mmd", "spectral_mmd", "learned_feature_mmd", "classifier_auc_mean"),
) -> list[Path]:
    df = pd.read_csv(csv_path)
    out_dir = Path(output_dir)
    paths = []
    for metric in metrics:
        if metric in df.columns:
            paths.append(bar_plot_metric(df, metric, out_dir / f"{metric}.png"))
    return paths
