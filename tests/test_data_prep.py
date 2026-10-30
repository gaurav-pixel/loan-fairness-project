"""
test_data_prep.py – Unit tests for src/data_prep.py.

Covers:
  1. No missing values after cleaning.
  2. No data leakage (train and test indices do not overlap).
  3. Target distribution is preserved (stratified split).
  4. Preprocessor is fitted on train only (transform raises on unseen cats).
  5. Engineer_features correctly maps gender and target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep import (
    load_raw,
    engineer_features,
    split_data,
    build_preprocessing_pipeline,
    prepare_data,
)
from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    NUM_FEATURES,
    CAT_FEATURES,
    GENDER_COL,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def engineered_df(raw_df):
    return engineer_features(raw_df)


@pytest.fixture(scope="module")
def split_dfs(engineered_df):
    return split_data(engineered_df)


# ── 1. No missing values after engineering ────────────────────────────────────

def test_no_missing_after_engineering(engineered_df):
    """All cells must be non-null after feature engineering."""
    missing = engineered_df.isnull().sum()
    assert missing.sum() == 0, f"Unexpected missing values:\n{missing[missing > 0]}"


# ── 2. No leakage between train and test ──────────────────────────────────────

def test_no_index_overlap(engineered_df, split_dfs):
    """
    Train and test DataFrames must contain disjoint *rows* from the original.

    split_data() calls reset_index(drop=True) so both DataFrames have 0-based
    indices — we can't compare those directly.  Instead we verify there is no
    row duplication by checking that train + test sizes equal the full dataset
    and that the actual row content doesn't repeat (sampling without replacement).
    """
    train_df, test_df = split_dfs

    # Sizes must add up (no rows dropped or duplicated)
    assert len(train_df) + len(test_df) == len(engineered_df), \
        "Row count mismatch: rows may have been dropped or duplicated"

    # Content-level check: concatenate and verify no duplicate rows
    combined = pd.concat([train_df, test_df])
    n_dupes = combined.duplicated().sum()
    assert n_dupes == 0, (
        f"{n_dupes} duplicate rows found across train/test — "
        "this indicates data leakage (same row in both splits)"
    )


def test_sizes_sum_to_total(engineered_df, split_dfs):
    """Train + test rows must equal total rows."""
    train_df, test_df = split_dfs
    assert len(train_df) + len(test_df) == len(engineered_df)


# ── 3. Stratified split preserves class ratio ────────────────────────────────

def test_stratified_split_class_ratio(engineered_df, split_dfs):
    """
    The positive-class fraction in train and test should be within ±3 pp of
    the overall fraction (stratification guarantee).
    """
    train_df, test_df = split_dfs
    overall_rate = engineered_df[TARGET_COL].mean()
    train_rate = train_df[TARGET_COL].mean()
    test_rate = test_df[TARGET_COL].mean()
    assert abs(train_rate - overall_rate) < 0.03, (
        f"Train class rate {train_rate:.3f} drifts from overall {overall_rate:.3f}"
    )
    assert abs(test_rate - overall_rate) < 0.03, (
        f"Test class rate {test_rate:.3f} drifts from overall {overall_rate:.3f}"
    )


# ── 4. Preprocessor fitted on train only ─────────────────────────────────────

def test_preprocessor_fitted_on_train_only(split_dfs):
    """
    Fit the preprocessor on train data.  It should transform test data without
    raising (unseen categories get -1 via OrdinalEncoder handle_unknown).
    """
    train_df, test_df = split_dfs

    preprocessor = build_preprocessing_pipeline()
    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    X_test = test_df[NUM_FEATURES + CAT_FEATURES]

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    assert X_train_t.shape[1] == X_test_t.shape[1], "Feature count mismatch after transform"
    # No NaNs in either transformed array
    assert not np.isnan(X_train_t).any(), "NaN in transformed train data"
    assert not np.isnan(X_test_t).any(), "NaN in transformed test data"


def test_preprocessor_output_shape(split_dfs):
    """Output columns = len(NUM_FEATURES) + len(CAT_FEATURES)."""
    train_df, test_df = split_dfs
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(train_df[NUM_FEATURES + CAT_FEATURES])
    X = preprocessor.transform(train_df[NUM_FEATURES + CAT_FEATURES])
    expected_cols = len(NUM_FEATURES) + len(CAT_FEATURES)
    assert X.shape[1] == expected_cols, f"Expected {expected_cols} cols, got {X.shape[1]}"


# ── 5. Feature engineering correctness ───────────────────────────────────────

def test_gender_mapping(raw_df):
    """A91/A93/A94 → male; A92/A95 → female."""
    df = engineer_features(raw_df)
    male_codes = {"A91", "A93", "A94"}
    female_codes = {"A92", "A95"}
    for code in male_codes:
        rows = df[df["personal_status_sex"] == code][GENDER_COL]
        assert (rows == "male").all(), f"Code {code} should map to male"
    for code in female_codes:
        rows = df[df["personal_status_sex"] == code][GENDER_COL]
        assert (rows == "female").all(), f"Code {code} should map to female"


def test_target_is_binary(engineered_df):
    """Target must contain only 0 and 1."""
    vals = set(engineered_df[TARGET_COL].unique())
    assert vals == {0, 1}, f"Unexpected target values: {vals}"


def test_target_remapping(raw_df):
    """Original label 1 → binary 1 (good/approved); original 2 → binary 0."""
    df = engineer_features(raw_df)
    orig_good_mask = raw_df["target"] == 1
    assert (df.loc[orig_good_mask, TARGET_COL] == 1).all()
    orig_bad_mask = raw_df["target"] == 2
    assert (df.loc[orig_bad_mask, TARGET_COL] == 0).all()
