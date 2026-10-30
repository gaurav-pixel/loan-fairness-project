"""
test_model.py – Smoke tests for model training and prediction.

Verifies that:
  1. Both models produce probabilities in [0, 1] for every test sample.
  2. Hard predictions are binary (0 or 1 only).
  3. Saved model can be reloaded and produces identical predictions.
  4. Preprocessor output has the expected number of features.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep import prepare_data
from src.train import build_logistic_regression, build_random_forest, train_model
from src.config import LOGREG_PATH, RF_PATH, NUM_FEATURES, CAT_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    """Prepare data once for all tests in this module (expensive I/O)."""
    return prepare_data(save=False)


@pytest.fixture(scope="module")
def trained_logreg(data):
    X_train, _, y_train, *_ = data
    return train_model(build_logistic_regression(), X_train, y_train)


@pytest.fixture(scope="module")
def trained_rf(data):
    X_train, _, y_train, *_ = data
    return train_model(build_random_forest(), X_train, y_train)


# ── 1. Probability bounds ─────────────────────────────────────────────────────

class TestProbabilityBounds:
    def test_logreg_probas_in_unit_interval(self, trained_logreg, data):
        _, X_test, _, y_test, *_ = data
        probas = trained_logreg.predict_proba(X_test)[:, 1]
        assert (probas >= 0.0).all(), "Probability < 0 found"
        assert (probas <= 1.0).all(), "Probability > 1 found"

    def test_rf_probas_in_unit_interval(self, trained_rf, data):
        _, X_test, _, y_test, *_ = data
        probas = trained_rf.predict_proba(X_test)[:, 1]
        assert (probas >= 0.0).all(), "Probability < 0 found"
        assert (probas <= 1.0).all(), "Probability > 1 found"

    def test_single_sample_proba(self, trained_rf, data):
        _, X_test, *_ = data
        proba = trained_rf.predict_proba(X_test[[0]])[:, 1][0]
        assert 0.0 <= proba <= 1.0, f"Single-sample proba out of range: {proba}"


# ── 2. Hard predictions are binary ───────────────────────────────────────────

class TestHardPredictions:
    def test_logreg_binary_output(self, trained_logreg, data):
        _, X_test, *_ = data
        preds = trained_logreg.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1}), f"Non-binary preds: {np.unique(preds)}"

    def test_rf_binary_output(self, trained_rf, data):
        _, X_test, *_ = data
        preds = trained_rf.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1}), f"Non-binary preds: {np.unique(preds)}"


# ── 3. Serialise / reload ─────────────────────────────────────────────────────

def test_model_roundtrip(trained_rf, data, tmp_path):
    """Saved and reloaded model must produce identical predictions."""
    _, X_test, *_ = data
    save_path = tmp_path / "rf_test.joblib"
    joblib.dump(trained_rf, save_path)
    reloaded = joblib.load(save_path)
    np.testing.assert_array_equal(
        trained_rf.predict(X_test),
        reloaded.predict(X_test),
        err_msg="Predictions differ after joblib round-trip",
    )


# ── 4. Preprocessor shape ────────────────────────────────────────────────────

def test_preprocessor_feature_count(data):
    """Transformed X must have exactly len(NUM+CAT) columns."""
    X_train, X_test, *_ = data
    expected = len(NUM_FEATURES) + len(CAT_FEATURES)
    assert X_train.shape[1] == expected, (
        f"Expected {expected} features, got {X_train.shape[1]}"
    )
    assert X_test.shape[1] == expected


# ── 5. Model accuracy sanity check ───────────────────────────────────────────

def test_rf_accuracy_above_chance(trained_rf, data):
    """RF should beat 50% accuracy on this non-trivial dataset."""
    from sklearn.metrics import accuracy_score
    _, X_test, _, y_test, *_ = data
    acc = accuracy_score(y_test, trained_rf.predict(X_test))
    assert acc > 0.55, f"RF accuracy suspiciously low: {acc:.4f}"
