"""
fairness.py – Per-group metrics, demographic parity, and equal opportunity.

Uses fairlearn's MetricFrame for concise grouped computation.  Raw numpy/
pandas implementations are also provided so the module works without
fairlearn installed (and to support unit-testing with controlled inputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.config import FIGURES_DIR, SENSITIVE_ATTRS

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Low-level fairness scalars ─────────────────────────────────────────────

def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    Demographic Parity Difference (DPD).

    DPD = max_group(approval_rate) - min_group(approval_rate)

    A value of 0 means every group has the same approval rate.
    A common rule-of-thumb threshold is 0.10 (10 pp gap).
    """
    groups = np.unique(sensitive)
    rates = {g: y_pred[sensitive == g].mean() for g in groups}
    return float(max(rates.values()) - min(rates.values()))


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    Equal Opportunity Difference (EOD).

    EOD = max_group(TPR) - min_group(TPR)

    TPR = recall = P(ŷ=1 | y=1).
    A value close to 0 means qualified applicants of every group have the same
    chance of being approved.
    """
    groups = np.unique(sensitive)
    tprs: dict[str, float] = {}
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            tprs[g] = float("nan")
        else:
            tprs[g] = float(y_pred[mask].mean())
    valid = [v for v in tprs.values() if not np.isnan(v)]
    if len(valid) < 2:
        return float("nan")
    return float(max(valid) - min(valid))


def approval_rate_by_group(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> pd.Series:
    """Return Series of approval rates indexed by group label."""
    df = pd.DataFrame({"pred": y_pred, "group": sensitive})
    return df.groupby("group")["pred"].mean().rename("approval_rate")


def per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> pd.DataFrame:
    """
    Build a per-group metrics table with: approval_rate, TPR, FPR, n_samples.
    """
    rows = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        yt = y_true[mask]
        yp = y_pred[mask]
        n = mask.sum()
        ar = yp.mean()
        tpr = yp[yt == 1].mean() if (yt == 1).sum() > 0 else float("nan")
        fpr = yp[yt == 0].mean() if (yt == 0).sum() > 0 else float("nan")
        rows.append(
            {
                "group": g,
                "n": int(n),
                "approval_rate": round(float(ar), 4),
                "TPR (recall)": round(float(tpr), 4) if not np.isnan(tpr) else "N/A",
                "FPR": round(float(fpr), 4) if not np.isnan(fpr) else "N/A",
            }
        )
    return pd.DataFrame(rows).set_index("group")


# ── 2. fairlearn MetricFrame (optional but recommended) ───────────────────────

def fairlearn_metric_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_series: pd.Series,
) -> "fairlearn.metrics.MetricFrame":  # type: ignore[name-defined]
    """
    Build a fairlearn MetricFrame for rich grouped reporting.
    Raises ImportError if fairlearn is not installed.
    """
    try:
        from fairlearn.metrics import (
            MetricFrame,
            selection_rate,
            true_positive_rate,
            false_positive_rate,
        )
        from sklearn.metrics import accuracy_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fairlearn is required for this function.") from exc

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series,
    )
    return mf


# ── 3. Audit summary ──────────────────────────────────────────────────────────

def audit_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    attr_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run per-group metrics for every sensitive attribute.

    Returns a dict {attr_name: metrics_dataframe}.
    """
    if attr_cols is None:
        attr_cols = SENSITIVE_ATTRS

    report: dict[str, pd.DataFrame] = {}
    for attr in attr_cols:
        if attr not in test_df.columns:
            print(f"  Skipping '{attr}' – not in test_df")
            continue
        sensitive = test_df[attr].astype(str).values
        table = per_group_metrics(y_true, y_pred, sensitive)
        dpd = demographic_parity_difference(y_pred, sensitive)
        eod = equal_opportunity_difference(y_true, y_pred, sensitive)
        print(f"\n── {attr.upper()} ───────────────────────────────────────")
        print(table.to_string())
        print(f"  Demographic Parity Difference : {dpd:.4f}")
        print(f"  Equal Opportunity Difference  : {eod:.4f}")
        report[attr] = table
    return report


# ── 4. Visualisations ─────────────────────────────────────────────────────────

def plot_approval_rates(
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    attr_cols: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """Bar chart of approval rate per group for each sensitive attribute."""
    if attr_cols is None:
        attr_cols = SENSITIVE_ATTRS

    valid_attrs = [a for a in attr_cols if a in test_df.columns]
    ncols = len(valid_attrs)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, attr in zip(axes, valid_attrs):
        sensitive = test_df[attr].astype(str).values
        rates = approval_rate_by_group(y_pred, sensitive).sort_values()
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(rates)))
        rates.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.6)
        ax.set_title(f"Approval rate by {attr}", fontsize=11)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)
        # Add value labels
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1%}",
                (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.01),
                ha="center",
                fontsize=9,
            )

    fig.suptitle("Fairness Audit – Approval Rates by Sensitive Group", y=1.02)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "approval_rates_by_group.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    return fig
