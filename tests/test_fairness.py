"""
test_fairness.py – Unit tests for src/fairness.py metric functions.

Uses small synthetic examples with known ground-truth outputs so the tests
do not depend on the full dataset or model training.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fairness import (
    demographic_parity_difference,
    equal_opportunity_difference,
    approval_rate_by_group,
    per_group_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_parity_data():
    """Both groups have 50% approval → DPD = 0."""
    sensitive = np.array(["A", "A", "B", "B"])
    y_pred    = np.array([1,    0,    1,    0])
    y_true    = np.array([1,    0,    1,    0])
    return y_true, y_pred, sensitive


@pytest.fixture
def high_disparity_data():
    """Group A: 100% approval, Group B: 0% approval → DPD = 1.0."""
    sensitive = np.array(["A", "A", "B", "B"])
    y_pred    = np.array([1,    1,    0,    0])
    y_true    = np.array([1,    1,    0,    0])
    return y_true, y_pred, sensitive


@pytest.fixture
def partial_disparity_data():
    """Group A: 75% approval, Group B: 25% approval → DPD = 0.50."""
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    y_pred    = np.array([1,    1,    1,    0,    1,    0,    0,    0])
    y_true    = np.array([1,    1,    0,    0,    1,    1,    0,    0])
    return y_true, y_pred, sensitive


# ── Demographic Parity Difference ────────────────────────────────────────────

class TestDemographicParityDifference:
    def test_zero_when_equal_rates(self, perfect_parity_data):
        _, y_pred, sensitive = perfect_parity_data
        assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(0.0)

    def test_one_when_fully_disparate(self, high_disparity_data):
        _, y_pred, sensitive = high_disparity_data
        assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(1.0)

    def test_partial_disparity(self, partial_disparity_data):
        _, y_pred, sensitive = partial_disparity_data
        dpd = demographic_parity_difference(y_pred, sensitive)
        assert dpd == pytest.approx(0.50, abs=1e-6)

    def test_non_negative(self):
        sensitive = np.array(["X"] * 10 + ["Y"] * 10)
        y_pred = np.random.default_rng(0).integers(0, 2, size=20)
        assert demographic_parity_difference(y_pred, sensitive) >= 0.0

    def test_single_group_returns_zero(self):
        sensitive = np.array(["A"] * 8)
        y_pred    = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(0.0)


# ── Equal Opportunity Difference ─────────────────────────────────────────────

class TestEqualOpportunityDifference:
    def test_zero_when_equal_tpr(self, perfect_parity_data):
        y_true, y_pred, sensitive = perfect_parity_data
        # Both groups have TPR = 1.0 (every positive is caught)
        assert equal_opportunity_difference(y_true, y_pred, sensitive) == pytest.approx(0.0)

    def test_partial_eod(self):
        """
        Group A positives: [1, 1] → predicted [1, 0] → TPR = 0.5
        Group B positives: [1, 1] → predicted [1, 1] → TPR = 1.0
        EOD = 1.0 - 0.5 = 0.5
        """
        y_true    = np.array([1, 1, 1, 1])
        y_pred    = np.array([1, 0, 1, 1])
        sensitive = np.array(["A", "A", "B", "B"])
        eod = equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eod == pytest.approx(0.5, abs=1e-6)

    def test_nan_when_group_has_no_positives(self):
        """If a group has no positive labels, EOD should be NaN."""
        y_true    = np.array([0, 0, 1, 1])
        y_pred    = np.array([0, 0, 1, 1])
        sensitive = np.array(["A", "A", "B", "B"])
        eod = equal_opportunity_difference(y_true, y_pred, sensitive)
        # Group A has no positives → nan; only one valid group → nan returned
        assert np.isnan(eod)

    def test_non_negative_when_valid(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 40)
        y_pred = rng.integers(0, 2, 40)
        sensitive = np.where(rng.integers(0, 2, 40) == 0, "A", "B")
        eod = equal_opportunity_difference(y_true, y_pred, sensitive)
        if not np.isnan(eod):
            assert eod >= 0.0


# ── Approval rate by group ────────────────────────────────────────────────────

class TestApprovalRateByGroup:
    def test_values(self, partial_disparity_data):
        _, y_pred, sensitive = partial_disparity_data
        rates = approval_rate_by_group(y_pred, sensitive)
        assert rates["A"] == pytest.approx(0.75, abs=1e-6)
        assert rates["B"] == pytest.approx(0.25, abs=1e-6)

    def test_returns_series(self, perfect_parity_data):
        import pandas as pd
        _, y_pred, sensitive = perfect_parity_data
        result = approval_rate_by_group(y_pred, sensitive)
        assert isinstance(result, pd.Series)


# ── Per-group metrics DataFrame ───────────────────────────────────────────────

class TestPerGroupMetrics:
    def test_shape(self, partial_disparity_data):
        y_true, y_pred, sensitive = partial_disparity_data
        df = per_group_metrics(y_true, y_pred, sensitive)
        assert set(df.index) == {"A", "B"}
        assert "approval_rate" in df.columns
        assert "TPR (recall)" in df.columns
        assert "FPR" in df.columns

    def test_n_sums_to_total(self, partial_disparity_data):
        y_true, y_pred, sensitive = partial_disparity_data
        df = per_group_metrics(y_true, y_pred, sensitive)
        assert df["n"].sum() == len(y_true)
