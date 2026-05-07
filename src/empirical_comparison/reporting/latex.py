from __future__ import annotations

import pandas as pd


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(index=False, caption=caption, label=label, escape=False)


def format_mean_std(mean, std=None, precision: int = 4) -> str:
    if pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        return f"{float(mean):.{precision}f}" if isinstance(mean, (float, int)) else str(mean)
    return f"{float(mean):.{precision}f} $\\pm$ {float(std):.{precision}f}"
