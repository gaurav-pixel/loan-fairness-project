"""
evaluate.py – Classification metrics, confusion matrix, ROC curve, and
error analysis (false approvals / false rejections).

All plotting functions save figures to FIGURES_DIR and also return the
matplotlib Figure so notebooks can display them inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

from src.config import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Metrics table ──────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Return a dict of key classification metrics for a single model.

    Parameters
    ----------
    y_true  : ground-truth binary labels
    y_pred  : hard predictions (0/1)
    y_proba : predicted probability of the positive class (approved=1)
    model_name : label used in the returned dict
    """
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a formatted comparison table from a list of metric dicts.

    Example
    -------
    >>> compare_models([compute_metrics(..., 'LR'), compute_metrics(..., 'RF')])
    """
    df = pd.DataFrame(results).set_index("model")
    return df


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    save: bool = True,
) -> plt.Figure:
    """Plot and optionally save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
        ax=ax,
    )
    # Also overlay raw counts
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"n={cm[i,j]}",
                ha="center",
                va="center",
                fontsize=8,
                color="grey",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    return fig


# ── 3. ROC curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(
    models_probas: list[tuple[str, np.ndarray, np.ndarray]],
    save: bool = True,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple models on one axes.

    Parameters
    ----------
    models_probas : list of (name, y_true, y_proba) tuples
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    for name, y_true, y_proba in models_probas:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "roc_curves.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    return fig


# ── 4. Error analysis ─────────────────────────────────────────────────────────

def error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify false approvals (FP) and false rejections (FN) and compare their
    feature distributions against correct predictions.

    Returns
    -------
    false_approvals_df  : rows where model predicted approved but true = rejected
    false_rejections_df : rows where model predicted rejected but true = approved
    """
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    false_approvals = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    false_rejections = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]

    if feature_cols is None:
        feature_cols = ["age", "credit_amount", "duration_months", "gender", "foreign_worker"]

    print("── False Approvals (FP) ──────────────────────────────────────────")
    print(f"  Count: {len(false_approvals)}")
    for col in feature_cols:
        if col in false_approvals.columns:
            if false_approvals[col].dtype in [float, int, np.float64, np.int64]:
                print(f"  {col}: mean={false_approvals[col].mean():.2f}  "
                      f"(overall mean={df[col].mean():.2f})")
            else:
                print(f"  {col}: {false_approvals[col].value_counts(normalize=True).to_dict()}")

    print("\n── False Rejections (FN) ─────────────────────────────────────────")
    print(f"  Count: {len(false_rejections)}")
    for col in feature_cols:
        if col in false_rejections.columns:
            if false_rejections[col].dtype in [float, int, np.float64, np.int64]:
                print(f"  {col}: mean={false_rejections[col].mean():.2f}  "
                      f"(overall mean={df[col].mean():.2f})")
            else:
                print(f"  {col}: {false_rejections[col].value_counts(normalize=True).to_dict()}")

    return false_approvals, false_rejections


def plot_error_demographics(
    false_approvals: pd.DataFrame,
    false_rejections: pd.DataFrame,
    col: str = "gender",
    save: bool = True,
) -> plt.Figure:
    """Bar chart comparing error type distribution across a demographic column."""
    fa_counts = false_approvals[col].value_counts(normalize=True).rename("False Approval")
    fr_counts = false_rejections[col].value_counts(normalize=True).rename("False Rejection")
    combo = pd.concat([fa_counts, fr_counts], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    combo.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", width=0.6)
    ax.set_title(f"Error type distribution by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Proportion within error type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"error_demographics_{col}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    return fig
