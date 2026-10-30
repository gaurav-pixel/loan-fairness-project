"""
data_prep.py – Load, clean, encode, and split the German Credit dataset.

Design principles
-----------------
* All transforms are fitted on the training set only, then applied to test.
* A scikit-learn Pipeline encapsulates imputation + encoding to prevent leakage.
* A fixed random seed makes every split reproducible.
* Class-imbalance is handled via class_weight='balanced' in models, not here,
  so the raw label distribution is preserved for honest evaluation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    RAW_DATA_PATH,
    RAW_COLUMNS,
    TARGET_COL,
    GENDER_COL,
    AGE_GROUP_COL,
    AGE_BINS,
    AGE_LABELS,
    CAT_FEATURES,
    NUM_FEATURES,
    PROCESSED_DIR,
    PROCESSED_TRAIN,
    PROCESSED_TEST,
    PIPELINE_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)


# ── 1. Raw data loader ─────────────────────────────────────────────────────────

def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the space-separated german.data file and attach column headers."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    return df


# ── 2. Feature engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive helper columns from raw data.

    * ``gender``    – binary (male / female) extracted from personal_status_sex.
    * ``age_group`` – ordinal age bucket for group-level fairness analysis.
    * ``target``    – remap 1→1 (good/approved) and 2→0 (bad/rejected) so that
                      1 is always the positive class (approved).

    The original ``personal_status_sex`` column is kept; it will be dropped only
    in the mitigation experiment if chosen as a proxy feature to remove.
    """
    df = df.copy()

    # Gender: codes A91/A93/A94 are male; A92/A95 are female
    male_codes = {"A91", "A93", "A94"}
    df[GENDER_COL] = df["personal_status_sex"].apply(
        lambda x: "male" if x in male_codes else "female"
    )

    # Age group
    df[AGE_GROUP_COL] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)

    # Remap target: 1 (Good) → 1 (approved), 2 (Bad) → 0 (rejected)
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)

    return df


# ── 3. Missing-value / outlier check ──────────────────────────────────────────

def report_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column (should be 0 for German Credit)."""
    return df.isnull().sum()


def clip_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Winsorise numerical columns at the 1st and 99th percentiles.
    Fitted boundaries are derived from ``df`` itself (call only on training data).
    """
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
    return df


# ── 4. Train / test split ─────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split so each fold preserves the ~70/30 class ratio.
    Returns (train_df, test_df).
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── 5. sklearn Pipeline ───────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    cat_features: list[str] = CAT_FEATURES,
    num_features: list[str] = NUM_FEATURES,
) -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer that:
    * Imputes missing numerics with the median.
    * Scales numerics with StandardScaler.
    * Imputes missing categoricals with the most-frequent value.
    * Ordinal-encodes categoricals (handles_unknown='use_encoded_value').

    Fitting must happen on training data only.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",  # drops sensitive columns not in num/cat lists
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return ordered feature names after transformation."""
    num_names = preprocessor.transformers_[0][2]
    cat_names = list(preprocessor.transformers_[1][2])
    return list(num_names) + cat_names


# ── 6. End-to-end prepare function ────────────────────────────────────────────

def prepare_data(
    raw_path: Path = RAW_DATA_PATH,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load → engineer → split → fit preprocessor on train →
    transform both sets.

    Returns
    -------
    X_train, X_test : np.ndarray  (preprocessed feature matrices)
    y_train, y_test : np.ndarray  (binary labels)
    preprocessor    : fitted ColumnTransformer
    train_df        : raw training DataFrame (includes sensitive cols)
    test_df         : raw test DataFrame     (includes sensitive cols)
    """
    df = load_raw(raw_path)
    df = engineer_features(df)

    train_df, test_df = split_data(df)

    # Clip outliers using training-set percentiles only
    train_df = clip_outliers(train_df, NUM_FEATURES)
    train_clip_bounds = {
        col: (train_df[col].min(), train_df[col].max()) for col in NUM_FEATURES
    }
    for col in NUM_FEATURES:
        lo, hi = train_clip_bounds[col]
        test_df[col] = test_df[col].clip(lo, hi)

    X_train = train_df[NUM_FEATURES + CAT_FEATURES]
    y_train = train_df[TARGET_COL].values
    X_test = test_df[NUM_FEATURES + CAT_FEATURES]
    y_test = test_df[TARGET_COL].values

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_TRAIN, index=False)
        test_df.to_csv(PROCESSED_TEST, index=False)
        joblib.dump(preprocessor, PIPELINE_PATH)
        print(f"Preprocessor saved → {PIPELINE_PATH}")

    return X_train_t, X_test_t, y_train, y_test, preprocessor, train_df, test_df


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep, tr_df, te_df = prepare_data()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Class balance (train): {pd.Series(y_tr).value_counts(normalize=True).to_dict()}")
