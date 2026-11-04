"""Model training and evaluation shared by notebook and agents."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target] + [c for c in df.columns if c.endswith("_id")])
    y = df[target]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )


def candidate_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": Pipeline([("pre", _preprocessor(X)),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": Pipeline([("pre", _preprocessor(X)),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                  class_weight="balanced", n_jobs=-1, random_state=rs))]),
        "gradient_boosting": Pipeline([("pre", _preprocessor(X)),
                                       ("clf", GradientBoostingClassifier(random_state=rs))]),
    }


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(model: Pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "gini": round(2 * roc_auc_score(y_test, proba) - 1, 4),
        "ks": round(ks_statistic(y_test, proba), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compare_models(df: pd.DataFrame, target: str) -> dict:
    """Train all candidates, return metrics + the fitted best model and split."""
    X_tr, X_te, y_tr, y_te = split(df, target)
    results, fitted = {}, {}
    for name, pipe in candidate_models(X_tr).items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        m = evaluate(pipe, X_te, y_te)
        m["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
        m["cv_auc_std"] = round(float(cv_auc.std()), 4)
        results[name] = m
        fitted[name] = pipe
    best = max(results, key=lambda k: results[k]["roc_auc"])
    return {"results": results, "best": best, "fitted": fitted,
            "split": (X_tr, X_te, y_tr, y_te)}


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def evaluation_plots(model: Pipeline, X_test, y_test, label: str) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    figs = {}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, proba):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax[0].legend()
    p, r, _ = precision_recall_curve(y_test, proba)
    ax[1].plot(r, p, label=f"AP={average_precision_score(y_test, proba):.3f}")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax[1].legend()
    figs["roc_pr"] = _save(fig, f"{label}_roc_pr")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    frac, mean = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    ax.plot(mean, frac, "o-")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="Calibration", xlabel="Predicted", ylabel="Observed")
    figs["calibration"] = _save(fig, f"{label}_calibration")

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    s.plot.barh(ax=ax, color="#4C78A8")
    ax.set_title("Permutation importance (ΔAUC)")
    figs["importance"] = _save(fig, f"{label}_importance")
    figs["_importance_values"] = s.sort_values(ascending=False).round(4).to_dict()
    return figs


def segment_performance(model: Pipeline, X_test, y_test, column: str) -> pd.DataFrame:
    proba = model.predict_proba(X_test)[:, 1]
    d = pd.DataFrame({"seg": X_test[column].values, "y": y_test.values, "p": proba})
    rows = []
    for seg, g in d.groupby("seg"):
        auc = roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan
        rows.append({column: seg, "n": len(g), "actual_rate": round(g.y.mean(), 4),
                     "mean_pred": round(g.p.mean(), 4), "auc": round(auc, 4)})
    return pd.DataFrame(rows)
