"""
train.py – Train baseline (Logistic Regression) and improved (Random Forest)
models, then save each one with joblib.

Run directly:  python -m src.train
"""

from __future__ import annotations

import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    LOGREG_PATH,
    RF_PATH,
    BEST_MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.data_prep import prepare_data


# ── Model factories ────────────────────────────────────────────────────────────

def build_logistic_regression(random_state: int = RANDOM_SEED) -> LogisticRegression:
    """
    Logistic Regression baseline.
    * class_weight='balanced' compensates for the ~70/30 class imbalance.
    * max_iter=1000 ensures convergence on the ordinal-encoded features.
    """
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )


def build_random_forest(random_state: int = RANDOM_SEED) -> RandomForestClassifier:
    """
    Random Forest improved model.
    * n_estimators=400 gives stable OOB estimates without excessive runtime.
    * max_depth=8 prevents overfitting on 800 training samples.
    * class_weight='balanced_subsample' re-weights each bootstrap independently.
    """
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )


# ── Training helpers ───────────────────────────────────────────────────────────

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Fit ``model`` on training data and return the fitted estimator."""
    model.fit(X_train, y_train)
    return model


def save_model(model, path) -> None:
    """Persist a fitted model to disk via joblib."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


# ── Main training routine ──────────────────────────────────────────────────────

def run_training() -> tuple:
    """
    End-to-end training:
    1. Prepare data (load, engineer, split, preprocess).
    2. Train Logistic Regression and Random Forest.
    3. Save both models; copy the better one to BEST_MODEL_PATH.

    Returns
    -------
    logreg, rf : fitted models
    X_train, X_test, y_train, y_test : arrays
    preprocessor, train_df, test_df : for downstream use
    """
    print("── Data preparation ───────────────────────────────────────────")
    X_train, X_test, y_train, y_test, preprocessor, train_df, test_df = prepare_data()
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}")

    print("\n── Training Logistic Regression ───────────────────────────────")
    logreg = train_model(build_logistic_regression(), X_train, y_train)
    save_model(logreg, LOGREG_PATH)

    print("\n── Training Random Forest ─────────────────────────────────────")
    rf = train_model(build_random_forest(), X_train, y_train)
    print(f"  OOB score: {rf.oob_score_:.4f}")
    save_model(rf, RF_PATH)

    # Pick the model with the higher ROC-AUC on test as the "best"
    from sklearn.metrics import roc_auc_score
    logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"\n  LogReg ROC-AUC (test): {logreg_auc:.4f}")
    print(f"  RF     ROC-AUC (test): {rf_auc:.4f}")

    best = rf if rf_auc >= logreg_auc else logreg
    save_model(best, BEST_MODEL_PATH)
    print(f"\n  Best model: {'Random Forest' if best is rf else 'Logistic Regression'}")

    return logreg, rf, X_train, X_test, y_train, y_test, preprocessor, train_df, test_df


if __name__ == "__main__":
    run_training()
