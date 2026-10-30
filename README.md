# Loan Approval Predictor with Fairness Audit and Explainability

> **Portfolio project** for an AI intern role.  
> Demonstrates data preparation, EDA, baseline modelling, error analysis,
> Responsible AI practices (fairness audit + mitigation), and SHAP explainability
> in a Streamlit demo.

---

## Problem Statement

Credit scoring models can inherit and amplify historical biases present in training
data. This project trains a loan approval model on the German Credit dataset,
measures fairness gaps across gender, age group, and foreign-worker status, applies
one mitigation technique, and explains individual predictions with SHAP.  The goal
is not just to build the most accurate model, but to build one whose decisions can
be audited, explained, and improved.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | German Credit Data (StatLog) |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Contributor | Prof. Dr. Hans Hofmann, Universität Hamburg |
| License | CC BY 4.0 |
| Size | 1 000 applicants × 20 features |
| Target | 1 = Good (approved), 0 = Bad (rejected) – 70 / 30 class split |
| Sensitive attrs | `gender` (derived), `age_group` (binned), `foreign_worker` |

See [`data/README.md`](data/README.md) for full column reference and privacy notice.

---

## Approach

```
raw data → feature engineering → stratified 80/20 split
         → sklearn Pipeline (impute + scale + encode, fit on train only)
         → Logistic Regression baseline
         → Random Forest (improved model, selected by ROC-AUC)
         → evaluation (accuracy, precision, recall, F1, ROC-AUC)
         → error analysis (false approvals / false rejections)
         → fairness audit (DPD, EOD per sensitive group via fairlearn)
         → mitigation (ExponentiatedGradient + DemographicParity constraint)
         → SHAP global summary + single-applicant waterfall
         → Streamlit demo
```

All transforms are fitted on the training set only.  A fixed random seed (42)
ensures reproducibility.

---

## Results

### Model comparison (test set, seed=42)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.72 | 0.79 | 0.84 | 0.81 | 0.77 |
| **Random Forest** | **0.76** | **0.83** | **0.85** | **0.84** | **0.81** |

**Why recall matters most**: A false rejection denies credit to a creditworthy
applicant.  This harms the individual (lost opportunity) and the lender (lost
revenue).  The German Credit cost matrix weights false approvals 5× more heavily,
but from a fairness perspective we also monitor false rejections carefully.

---

## Fairness Findings

### Before mitigation (Random Forest, gender attribute)

| Group | n | Approval rate | TPR |
|-------|---|---------------|-----|
| female | ~31 | ~65% | ~72% |
| male | ~169 | ~76% | ~86% |

- **Demographic Parity Difference**: ~11 pp
- **Equal Opportunity Difference**: ~14 pp

These gaps exist in the model's *predictions* and are not merely a reflection
of the original data distribution — they represent algorithmic disparity.

---

## Mitigation Tradeoff

Technique: **ExponentiatedGradient** with `DemographicParity` constraint (fairlearn)

| | Accuracy | Dem. Parity Diff | Equal Opp. Diff |
|--|----------|-----------------|----------------|
| Before | ~0.76 | ~0.11 | ~0.14 |
| After  | ~0.74 | ~0.06 | ~0.09 |

The mitigation reduced the gender disparity meaningfully but did not eliminate it.
Accuracy dropped ~2 pp — a real fairness-accuracy tradeoff.  Age-group and
foreign-worker gaps were not directly constrained and persist.

---

## SHAP – Top Features (Random Forest)

1. `checking_status` — account balance is the strongest approval signal
2. `credit_history` — repayment track record
3. `duration_months` — longer loans → higher rejection risk
4. `credit_amount` — larger loans → higher rejection risk
5. `savings_status` — savings reduce rejection risk
6. `age` — appears despite not being a target input; captured by tree splits

---

## Ethical Considerations

**Why a real lender should not deploy this model without human review:**

1. **Historical bias**: The dataset was collected in 1970s West Germany. The
   socioeconomic conditions, employment patterns, and gender roles of that era are
   embedded in the features and labels.

2. **Small dataset**: 1 000 samples is far too small for a production credit model.
   Groups like females (<31 in the test set) are drastically under-represented.

3. **Residual fairness gaps**: After mitigation, meaningful DPD and EOD gaps remain.
   Deploying a model with known disparities without further investigation may violate
   EU AI Act requirements (high-risk AI systems in credit scoring).

4. **Proxy features**: Features like `employment_since`, `foreign_worker`, and even
   `credit_amount` can act as proxies for protected characteristics.  Removing
   `personal_status_sex` alone is insufficient.

5. **Distributional shift**: Any model trained on historical data degrades as
   the applicant population changes. Ongoing monitoring is mandatory.

6. **Causal vs correlational**: The model identifies correlations, not causal credit
   risk factors. A variable may be predictive without being a fair basis for decisions.

---

## Limitations

- 1 000 samples is a toy scale; real credit models use millions of records.
- Only one mitigation technique was tested; in-processing (fairness constraints)
  is one of three broad families (pre-processing, in-processing, post-processing).
- SHAP values explain the model, not the data-generating process.
- The Streamlit demo uses a model serialised from one training run; a production
  system would need versioned models, A/B testing, and CI/CD.
- No temporal validation — the German Credit data has no time stamps, so the
  train/test split cannot simulate a real time-based deployment scenario.

---

## Setup and Run

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone and enter the project
git clone <your-repo-url> loan-fairness-project
cd loan-fairness-project

# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Provide the data

Copy `german.data` into `data/raw/`:

```bash
# Windows (adjust source path)
copy "path\to\german.data" data\raw\german.data
```

### Train the model

```bash
python -m src.train
```

This will:
1. Load and preprocess the data, saving splits to `data/processed/`.
2. Train Logistic Regression and Random Forest.
3. Save all models to `models/`.

### Run the tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Launch the Streamlit demo

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

### Run the notebooks

```bash
jupyter notebook notebooks/
```

Execute them in order: `01_eda.ipynb` → `02_modeling.ipynb` → `03_fairness_and_explainability.ipynb`.

---

## Project Structure

```
loan-fairness-project/
├── app.py                          # Streamlit demo (UI only)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── README.md                   # Dataset description, columns, license
│   ├── raw/german.data             # Source data (not committed to git)
│   └── processed/                  # Generated train.csv / test.csv
├── models/                         # Saved .joblib artefacts (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                # EDA
│   ├── 02_modeling.ipynb           # Training + evaluation
│   └── 03_fairness_and_explainability.ipynb
├── src/
│   ├── config.py                   # Paths, seeds, column names
│   ├── data_prep.py                # Load, clean, encode, split, pipeline
│   ├── train.py                    # Model training + serialisation
│   ├── evaluate.py                 # Metrics, confusion matrix, ROC, error analysis
│   ├── fairness.py                 # Per-group metrics, DPD, EOD
│   ├── mitigate.py                 # Reweighting / proxy removal + before/after
│   └── explain.py                  # SHAP global + single-applicant
├── reports/
│   ├── evaluation_report.md        # Written findings
│   └── figures/                    # Auto-generated charts
└── tests/
    ├── test_data_prep.py
    ├── test_fairness.py
    └── test_model.py
```

---

## Ideas for Future Work

1. **More mitigation techniques**: Post-processing threshold tuning per group
   (e.g., EqualizedOdds post-processor in fairlearn); adversarial debiasing.
2. **Better data**: Home Credit Default Risk dataset (Kaggle) — 300 k samples,
   richer temporal features, more balanced groups.
3. **Causal analysis**: Use a structural causal model to distinguish direct
   discrimination from indirect (proxy-based) discrimination.
4. **Model cards**: Auto-generate a Model Card (Mitchell et al., 2019) from the
   evaluation and fairness results.
5. **Intersectional fairness**: Audit the cross of gender × age group, not each
   in isolation.
6. **CI/CD pipeline**: GitHub Actions to retrain, run tests, and update the
   evaluation report on every data push.
7. **Calibration**: Platt scaling or isotonic regression so predicted probabilities
   are well-calibrated per group.
8. **Human-in-the-loop**: Build a review queue for near-threshold predictions
   (0.40–0.60 probability) that routes to human review.
