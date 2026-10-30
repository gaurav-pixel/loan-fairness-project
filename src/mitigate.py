"""
mitigate.py – Fairness mitigation via sample reweighting (ExponentiatedGradient
with DemographicParity) and proxy-feature removal.

Only ONE mitigation is applied in the main pipeline (reweighting), but both
approaches are implemented so they can be compared in the notebook.

IMPORTANT CAVEAT
----------------
Mitigation can reduce fairness gaps but rarely eliminates them entirely.
It also trades off accuracy.  The before/after table must be interpreted
with this context; this model must NOT be deployed without human review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    CAT_FEATURES,
    NUM_FEATURES,
    FIGURES_DIR,
)
from src.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    per_group_metrics,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Approach A – Reweighting via ExponentiatedGradient ─────────────────────

def train_mitigated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    random_state: int = RANDOM_SEED,
):
    """
    Train a fairness-constrained classifier using fairlearn's
    ExponentiatedGradient with a DemographicParity constraint.

    Falls back to a plain RandomForest if fairlearn is not available.

    Returns
    -------
    mitigated_clf : fitted classifier with .predict() and .predict_proba()
    """
    try:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity
        from sklearn.ensemble import RandomForestClassifier

        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
        constraint = DemographicParity()
        mitigated = ExponentiatedGradient(
            estimator=base,
            constraints=constraint,
            eps=0.05,          # allowed fairness violation
            max_iter=50,
        )
        mitigated.fit(X_train, y_train, sensitive_features=sensitive_train)
        print("Mitigated model trained with ExponentiatedGradient (DemographicParity).")
        return mitigated

    except ImportError:
        print("fairlearn not found – falling back to plain RandomForest.")
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        return rf


def save_mitigated_model(model, name: str = "mitigated_model") -> None:
    """Save the mitigated model to the models directory."""
    path = MODELS_DIR / f"{name}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Mitigated model saved → {path}")


# ── 2. Approach B – Proxy feature removal ─────────────────────────────────────

def remove_proxy_features(
    X_df: pd.DataFrame,
    proxy_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a copy of the feature DataFrame with proxy columns removed.
    Default proxy columns: personal_status_sex (directly encodes gender).

    Note: indirect proxies (e.g. zip code) cannot be removed this way.
    """
    if proxy_cols is None:
        proxy_cols = ["personal_status_sex"]
    cols_to_drop = [c for c in proxy_cols if c in X_df.columns]
    return X_df.drop(columns=cols_to_drop)


# ── 3. Before/after comparison table ─────────────────────────────────────────

def before_after_table(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    mitigated_pred: np.ndarray,
    sensitive: np.ndarray,
    attr_name: str = "gender",
) -> pd.DataFrame:
    """
    Build a table comparing accuracy and fairness metrics before and after
    mitigation for a single sensitive attribute.

    Returns a DataFrame with columns: [scenario, accuracy, DPD, EOD]
    """
    from sklearn.metrics import accuracy_score

    rows = []
    for label, y_pred in [("Before mitigation", baseline_pred),
                           ("After mitigation", mitigated_pred)]:
        acc = accuracy_score(y_true, y_pred)
        dpd = demographic_parity_difference(y_pred, sensitive)
        eod = equal_opportunity_difference(y_true, y_pred, sensitive)
        rows.append({
            "Scenario": label,
            "Accuracy": round(acc, 4),
            "Dem. Parity Diff": round(dpd, 4),
            "Equal Opp. Diff": round(eod, 4),
        })
    return pd.DataFrame(rows).set_index("Scenario")


def plot_before_after(
    table: pd.DataFrame,
    save: bool = True,
    attr_name: str = "gender",
):
    """Grouped bar chart of the before/after metrics table."""
    import matplotlib.pyplot as plt

    metrics = ["Accuracy", "Dem. Parity Diff", "Equal Opp. Diff"]
    fig, ax = plt.subplots(figsize=(8, 4))
    table[metrics].plot(kind="bar", ax=ax, colormap="Set1", edgecolor="black", width=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_title(f"Before vs After Mitigation – sensitive attr: {attr_name}", fontsize=11)
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"mitigation_before_after_{attr_name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    return fig
