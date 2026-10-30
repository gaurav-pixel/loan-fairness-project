"""
config.py – Central configuration: paths, seeds, column names, sensitive attributes.
All other modules import from here; no hardcoded values elsewhere.
"""

from pathlib import Path

# ── Repository root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "german.data"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_TRAIN = PROCESSED_DIR / "train.csv"
PROCESSED_TEST = PROCESSED_DIR / "test.csv"

# ── Model / artefact paths ─────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
LOGREG_PATH = MODELS_DIR / "logistic_regression.joblib"
RF_PATH = MODELS_DIR / "random_forest.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ── Reports ────────────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20

# ── Raw column names (space-separated, positional) ────────────────────────────
RAW_COLUMNS = [
    "checking_status",      # A1  – qualitative
    "duration_months",      # A2  – numerical
    "credit_history",       # A3  – qualitative
    "purpose",              # A4  – qualitative
    "credit_amount",        # A5  – numerical
    "savings_status",       # A6  – qualitative
    "employment_since",     # A7  – qualitative
    "installment_rate",     # A8  – numerical
    "personal_status_sex",  # A9  – qualitative  ← sensitive proxy
    "other_debtors",        # A10 – qualitative
    "residence_since",      # A11 – numerical
    "property",             # A12 – qualitative
    "age",                  # A13 – numerical    ← sensitive
    "other_installment",    # A14 – qualitative
    "housing",              # A15 – qualitative
    "existing_credits",     # A16 – numerical
    "job",                  # A17 – qualitative
    "liable_people",        # A18 – numerical
    "telephone",            # A19 – qualitative
    "foreign_worker",       # A20 – qualitative
    "target",               # 1 = Good (approved), 2 = Bad (rejected)
]

# ── Engineered column names ────────────────────────────────────────────────────
TARGET_COL = "target"           # binary: 1 = approved, 0 = rejected
GENDER_COL = "gender"           # derived from personal_status_sex
AGE_GROUP_COL = "age_group"     # derived from age

# ── Sensitive attributes used in fairness audit ────────────────────────────────
SENSITIVE_ATTRS = [GENDER_COL, AGE_GROUP_COL, "foreign_worker"]

# ── Categorical features (after renaming / engineering) ────────────────────────
CAT_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "other_debtors",
    "property",
    "other_installment",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# ── Numerical features ─────────────────────────────────────────────────────────
NUM_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "liable_people",
]

# ── Age-group cut-offs (years) ─────────────────────────────────────────────────
AGE_BINS = [0, 30, 45, 60, 120]
AGE_LABELS = ["<30", "30-45", "45-60", "60+"]
