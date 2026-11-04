"""EDA helpers shared by the notebook and the agent tools.

Every function returns plain data (dict / DataFrame / str path) so it can be
shown in a notebook or serialised back to an LLM agent.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def overview(df: pd.DataFrame, target: str) -> dict:
    """Shape, dtypes, memory, duplicates, target balance."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "target": target,
        "target_rate": round(float(df[target].mean()), 4),
        "target_counts": df[target].value_counts().to_dict(),
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame({"missing": m, "pct": (m / len(df) * 100).round(2)})
    return out[out.missing > 0].sort_values("pct", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number")
    s = num.describe().T
    s["skew"] = num.skew().round(2)
    return s.round(3)


def categorical_summary(df: pd.DataFrame, target: str) -> dict:
    out = {}
    for c in df.select_dtypes(exclude="number").columns:
        out[c] = (
            df.groupby(c)[target].agg(["count", "mean"]).rename(columns={"mean": "target_rate"}).round(4)
        ).to_dict("index")
    return out


def outlier_report(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """IQR-based outlier counts for numeric columns."""
    rows = []
    for c in df.select_dtypes("number").columns:
        if c == target or c.endswith("_id"):
            continue
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": c, "outliers": n, "pct": round(n / len(s) * 100, 2), "lower": lo, "upper": hi})
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)


def target_correlations(df: pd.DataFrame, target: str) -> pd.Series:
    num = df.select_dtypes("number").drop(columns=[c for c in df.columns if c.endswith("_id")], errors="ignore")
    return num.corr()[target].drop(target).sort_values(key=abs, ascending=False).round(3)


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_target_balance(df: pd.DataFrame, target: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 3))
    df[target].value_counts().sort_index().plot.bar(ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title(f"Target balance ({target})")
    return _save(fig, "target_balance")


def plot_distributions(df: pd.DataFrame, target: str) -> str:
    cols = [c for c in df.select_dtypes("number").columns if c != target and not c.endswith("_id")]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    for ax, c in zip(np.ravel(axes), cols):
        sns.histplot(data=df, x=c, hue=target, bins=30, stat="density", common_norm=False, ax=ax, legend=False)
        ax.set_title(c)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, "distributions")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    num = df.select_dtypes("number").drop(columns=[c for c in df.columns if c.endswith("_id")], errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix")
    return _save(fig, "correlation_heatmap")


def plot_target_rate_by_category(df: pd.DataFrame, target: str) -> str:
    cats = list(df.select_dtypes(exclude="number").columns)
    fig, axes = plt.subplots(1, max(len(cats), 1), figsize=(5 * max(len(cats), 1), 3.5))
    for ax, c in zip(np.atleast_1d(axes), cats):
        df.groupby(c)[target].mean().sort_values().plot.barh(ax=ax, color="#4C78A8")
        ax.set_title(f"{target} rate by {c}")
    fig.tight_layout()
    return _save(fig, "target_rate_by_category")


def run_full_eda(df: pd.DataFrame, target: str) -> dict:
    """Produce all EDA artefacts; returns a JSON-serialisable summary."""
    figs = {
        "target_balance": plot_target_balance(df, target),
        "distributions": plot_distributions(df, target),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "target_rate_by_category": plot_target_rate_by_category(df, target),
    }
    return {
        "overview": overview(df, target),
        "missing": missing_report(df).to_dict("index"),
        "numeric_summary": numeric_summary(df).to_dict("index"),
        "categorical_summary": categorical_summary(df, target),
        "outliers": outlier_report(df, target).to_dict("records"),
        "target_correlations": target_correlations(df, target).to_dict(),
        "figures": figs,
    }
