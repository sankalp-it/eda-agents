"""Strands tools wrapping the shared src/ modules.

Tools hold state (dataframe, fitted models) in a module-level session so the
LLM only passes small arguments and receives compact JSON summaries.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from strands import tool

from src import config, data, eda, modeling

_S: dict = {}  # session state


def _df():
    if "df" not in _S:
        _S["df"] = data.load_data()
    return _S["df"]


def _rel(p: str) -> str:
    """Figure path relative to the reports/ folder, for markdown embedding."""
    return str(Path(p).relative_to(config.REPORTS_DIR))


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


# ----------------------------- EDA tools ---------------------------------
@tool
def dataset_overview() -> str:
    """Return shape, dtypes, duplicate count and target balance of the dataset."""
    return _j(eda.overview(_df(), config.TARGET_COLUMN))


@tool
def data_quality_report() -> str:
    """Return missing-value percentages and IQR outlier counts per numeric column."""
    df = _df()
    return _j({
        "missing": eda.missing_report(df).to_dict("index"),
        "outliers": eda.outlier_report(df, config.TARGET_COLUMN).to_dict("records"),
    })


@tool
def feature_analysis() -> str:
    """Return numeric summary stats, categorical target rates and correlation with the target."""
    df, t = _df(), config.TARGET_COLUMN
    return _j({
        "numeric_summary": eda.numeric_summary(df).to_dict("index"),
        "categorical_target_rates": eda.categorical_summary(df, t),
        "target_correlations": eda.target_correlations(df, t).to_dict(),
    })


@tool
def generate_eda_figures() -> str:
    """Create EDA charts (target balance, distributions, correlation heatmap,
    target rate by category). Returns report-relative image paths."""
    df, t = _df(), config.TARGET_COLUMN
    figs = {
        "target_balance": eda.plot_target_balance(df, t),
        "distributions": eda.plot_distributions(df, t),
        "correlation_heatmap": eda.plot_correlation_heatmap(df),
        "target_rate_by_category": eda.plot_target_rate_by_category(df, t),
    }
    return _j({k: _rel(v) for k, v in figs.items()})


# -------------------------- Evaluation tools ------------------------------
@tool
def train_and_compare_models() -> str:
    """Train logistic regression, random forest and gradient boosting with 5-fold CV
    and hold-out evaluation. Returns metrics per model and the best model by ROC AUC."""
    cmp = modeling.compare_models(_df(), config.TARGET_COLUMN)
    _S["cmp"] = cmp
    return _j({"best_model": cmp["best"], "results": cmp["results"]})


@tool
def evaluate_best_model() -> str:
    """Create ROC/PR, calibration and permutation-importance charts for the best
    model and compute segment performance. Requires train_and_compare_models first."""
    if "cmp" not in _S:
        return "ERROR: call train_and_compare_models first."
    cmp = _S["cmp"]
    best = cmp["best"]
    _, X_te, _, y_te = cmp["split"]
    model = cmp["fitted"][best]
    plots = modeling.evaluation_plots(model, X_te, y_te, best)
    seg = {
        c: modeling.segment_performance(model, X_te, y_te, c).to_dict("records")
        for c in ("vehicle_type", "region") if c in X_te.columns
    }
    return _j({
        "best_model": best,
        "figures": {k: _rel(v) for k, v in plots.items() if not k.startswith("_")},
        "permutation_importance": plots["_importance_values"],
        "segment_performance": seg,
    })


@tool
def save_report(markdown: str) -> str:
    """Save the final Model Evaluation Report (markdown) to reports/.
    Embed figures with relative paths such as ![ROC](figures/xyz.png).

    Args:
        markdown: The complete report in markdown.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = config.REPORTS_DIR / f"model_evaluation_report_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    (config.REPORTS_DIR / "model_evaluation_report_latest.md").write_text(markdown, encoding="utf-8")
    return f"Saved report to {path}"


EDA_TOOLS = [dataset_overview, data_quality_report, feature_analysis, generate_eda_figures]
EVAL_TOOLS = [train_and_compare_models, evaluate_best_model]
