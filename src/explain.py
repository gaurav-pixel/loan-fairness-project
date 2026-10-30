"""
explain.py – SHAP-based global and single-applicant explanations.

* Global: TreeExplainer summary plot (beeswarm) for feature importance.
* Local : waterfall / force plot for a single applicant.

Both plotting functions save to FIGURES_DIR and return the figure so
they can be embedded in the Streamlit app or displayed in notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from src.config import FIGURES_DIR, NUM_FEATURES, CAT_FEATURES

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = NUM_FEATURES + CAT_FEATURES


# ── 1. Build SHAP explainer ───────────────────────────────────────────────────

def build_explainer(model, X_background: np.ndarray):
    """
    Build the appropriate SHAP explainer for the given model type.

    * TreeExplainer is used for tree-based models (RF, XGBoost).
    * LinearExplainer is used for Logistic Regression.
    * Falls back to KernelExplainer for anything else (slow).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required. Install with: pip install shap") from exc

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"):
        return shap.TreeExplainer(model)
    elif model_type == "LogisticRegression":
        return shap.LinearExplainer(model, X_background)
    else:
        # ExponentiatedGradient (fairlearn) – use underlying predictors
        # Fall back to KernelExplainer with a small background sample
        background = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )


def compute_shap_values(
    explainer,
    X: np.ndarray,
    model,
) -> np.ndarray:
    """
    Compute SHAP values.  For tree-based models returns an array of shape
    (n_samples, n_features); for classifiers with multiple outputs, returns
    the positive-class slice.
    """
    import shap

    shap_vals = explainer.shap_values(X)

    # TreeExplainer on a classifier returns a list [class0, class1]
    if isinstance(shap_vals, list):
        return shap_vals[1]  # positive class (approved)
    return shap_vals


# ── 2. Global summary plot ────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_display: int = 15,
    save: bool = True,
    model_name: str = "model",
) -> plt.Figure:
    """
    Beeswarm summary plot: each dot is a sample, position shows SHAP value,
    colour shows feature value (blue = low, red = high).
    """
    import shap

    if feature_names is None:
        feature_names = FEATURE_NAMES

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP Summary – {model_name}", y=1.01, fontsize=12)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"shap_summary_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    return fig


# ── 3. Single-applicant waterfall plot ───────────────────────────────────────

def plot_shap_waterfall(
    explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    sample_idx: int = 0,
    feature_names: list[str] | None = None,
    save: bool = True,
    label: str = "applicant",
) -> plt.Figure:
    """
    Waterfall plot for a single applicant: shows which features push the
    prediction towards approval or rejection relative to the base value.
    """
    import shap

    if feature_names is None:
        feature_names = FEATURE_NAMES

    try:
        # shap >= 0.40 Explanation object path
        exp = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value
            if not isinstance(explainer.expected_value, list)
            else explainer.expected_value[1],
            data=X[sample_idx],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(exp, show=False, max_display=12)
        fig = plt.gcf()
    except Exception:
        # Fallback: bar plot of this sample's SHAP values
        fig, ax = plt.subplots(figsize=(8, 5))
        sv = shap_values[sample_idx]
        order = np.argsort(np.abs(sv))[-12:][::-1]
        ax.barh(
            [feature_names[i] for i in order],
            sv[order],
            color=["#d73027" if v > 0 else "#4575b4" for v in sv[order]],
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value (impact on approval probability)")
        ax.set_title(f"SHAP – {label}")
        fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"shap_waterfall_{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    return fig


# ── 4. Convenience wrapper for the Streamlit app ─────────────────────────────

def explain_single(
    model,
    preprocessor,
    applicant_df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Compute SHAP values and prediction for a single-row applicant DataFrame.

    Parameters
    ----------
    model         : fitted sklearn / fairlearn model
    preprocessor  : fitted ColumnTransformer
    applicant_df  : single-row DataFrame with raw feature columns
    feature_names : list of feature names (default: NUM_FEATURES + CAT_FEATURES)

    Returns
    -------
    shap_vals  : 1-D array of SHAP values (one per feature)
    base_value : model base value (expected value)
    prob       : predicted approval probability
    """
    import shap

    if feature_names is None:
        feature_names = FEATURE_NAMES

    X_proc = preprocessor.transform(applicant_df[NUM_FEATURES + CAT_FEATURES])
    prob = float(model.predict_proba(X_proc)[0, 1])

    explainer = build_explainer(model, X_proc)
    sv = compute_shap_values(explainer, X_proc, model)

    base = (
        explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else (explainer.expected_value[1]
              if len(explainer.expected_value) > 1
              else explainer.expected_value[0])
    )

    return sv[0], float(base), prob
