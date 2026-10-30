"""
app.py – Streamlit demo for the Loan Approval Predictor.

UI only: loads the pre-trained model and preprocessing pipeline, collects
applicant details via a form, and displays:
  * Decision and approval probability
  * SHAP waterfall explanation
  * Fairness disclaimer

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path
import sys

# Allow imports from the project root when running with `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    BEST_MODEL_PATH,
    PIPELINE_PATH,
    NUM_FEATURES,
    CAT_FEATURES,
)
from src.explain import build_explainer, compute_shap_values

FEATURE_NAMES = NUM_FEATURES + CAT_FEATURES

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Demonstration only.** This tool was built for a portfolio project using the "
    "German Credit dataset.  It is **not** a real lending tool, has known fairness "
    "limitations, and must **not** be used to make actual lending decisions.  "
    "Any real deployment would require human review, regulatory compliance, and "
    "ongoing fairness auditing."
)

st.title("🏦 Loan Approval Predictor")
st.caption("German Credit Dataset · Fairness-Audited · SHAP Explanations")


# ── Load artefacts (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_artefacts():
    """Load model and preprocessor from disk; raise clear error if missing."""
    if not BEST_MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Model artefacts not found. Please run  `python -m src.train`  first."
        )
    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PIPELINE_PATH)
    return model, preprocessor


try:
    model, preprocessor = load_artefacts()
    artefacts_ok = True
except FileNotFoundError as exc:
    st.error(str(exc))
    artefacts_ok = False


# ── Applicant input form ──────────────────────────────────────────────────────

if artefacts_ok:
    st.subheader("Applicant Details")

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)

        with col1:
            checking_status = st.selectbox(
                "Checking account status",
                ["A11 (< 0 DM)", "A12 (0–200 DM)", "A13 (≥ 200 DM)", "A14 (none)"],
                help="Current balance of the applicant's checking account.",
            )
            duration_months = st.slider("Loan duration (months)", 4, 72, 24)
            credit_amount = st.number_input(
                "Credit amount (DM)", min_value=250, max_value=20000, value=3000, step=100
            )
            credit_history = st.selectbox(
                "Credit history",
                [
                    "A30 (no credits / all paid)",
                    "A31 (all at this bank paid)",
                    "A32 (existing credits paid)",
                    "A33 (delay in past)",
                    "A34 (critical / other credits)",
                ],
            )
            purpose = st.selectbox(
                "Purpose",
                [
                    "A40 (new car)", "A41 (used car)", "A42 (furniture)",
                    "A43 (radio/TV)", "A44 (appliances)", "A45 (repairs)",
                    "A46 (education)", "A48 (retraining)", "A49 (business)", "A410 (other)",
                ],
            )
            savings_status = st.selectbox(
                "Savings account",
                [
                    "A61 (< 100 DM)", "A62 (100–500 DM)", "A63 (500–1000 DM)",
                    "A64 (≥ 1000 DM)", "A65 (unknown/none)",
                ],
            )

        with col2:
            employment_since = st.selectbox(
                "Employed since",
                [
                    "A71 (unemployed)", "A72 (< 1 yr)", "A73 (1–4 yrs)",
                    "A74 (4–7 yrs)", "A75 (≥ 7 yrs)",
                ],
            )
            installment_rate = st.slider(
                "Installment rate (% of income)", 1, 4, 2,
                help="Monthly instalment as a % of disposable income."
            )
            age = st.slider("Age", 19, 75, 35)
            other_debtors = st.selectbox(
                "Other debtors / guarantors",
                ["A101 (none)", "A102 (co-applicant)", "A103 (guarantor)"],
            )
            property_type = st.selectbox(
                "Property",
                [
                    "A121 (real estate)", "A122 (building/life insurance)",
                    "A123 (car/other)", "A124 (none)",
                ],
            )
            other_installment = st.selectbox(
                "Other instalment plans",
                ["A141 (bank)", "A142 (stores)", "A143 (none)"],
            )
            housing = st.selectbox(
                "Housing", ["A151 (rent)", "A152 (own)", "A153 (free)"]
            )
            job = st.selectbox(
                "Job category",
                [
                    "A171 (unskilled non-resident)", "A172 (unskilled resident)",
                    "A173 (skilled employee)", "A174 (management/self-employed)",
                ],
            )
            residence_since = st.slider("Residence since (years)", 1, 4, 2)
            existing_credits = st.slider("Existing credits at this bank", 1, 4, 1)
            liable_people = st.slider("Number of dependants", 1, 2, 1)
            telephone = st.selectbox(
                "Telephone", ["A191 (none)", "A192 (registered)"]
            )
            foreign_worker = st.selectbox(
                "Foreign worker", ["A201 (yes)", "A202 (no)"]
            )

        submitted = st.form_submit_button("Predict", type="primary")

    # ── Prediction ─────────────────────────────────────────────────────────────

    def extract_code(s: str) -> str:
        """Extract the raw code from a display string like 'A11 (< 0 DM)'."""
        return s.split(" ")[0]

    if submitted:
        # Build applicant DataFrame
        applicant = {
            "checking_status": extract_code(checking_status),
            "duration_months": int(duration_months),
            "credit_history": extract_code(credit_history),
            "purpose": extract_code(purpose),
            "credit_amount": float(credit_amount),
            "savings_status": extract_code(savings_status),
            "employment_since": extract_code(employment_since),
            "installment_rate": int(installment_rate),
            "residence_since": int(residence_since),
            "age": int(age),
            "other_debtors": extract_code(other_debtors),
            "property": extract_code(property_type),
            "other_installment": extract_code(other_installment),
            "housing": extract_code(housing),
            "existing_credits": int(existing_credits),
            "job": extract_code(job),
            "liable_people": int(liable_people),
            "telephone": extract_code(telephone),
            "foreign_worker": extract_code(foreign_worker),
        }

        try:
            df_input = pd.DataFrame([applicant])

            # Validate all expected features are present
            missing_feats = [f for f in NUM_FEATURES + CAT_FEATURES if f not in df_input.columns]
            if missing_feats:
                st.error(f"Missing features: {missing_feats}")
                st.stop()

            X_proc = preprocessor.transform(df_input[NUM_FEATURES + CAT_FEATURES])
            proba = float(model.predict_proba(X_proc)[0, 1])
            decision = "✅ Approved" if proba >= 0.5 else "❌ Rejected"

            st.divider()
            col_d, col_p = st.columns(2)
            col_d.metric("Decision", decision)
            col_p.metric("Approval probability", f"{proba:.1%}")

            # ── SHAP explanation ───────────────────────────────────────────────
            st.subheader("Why this decision?")
            with st.spinner("Computing SHAP explanation…"):
                try:
                    explainer = build_explainer(model, X_proc)
                    sv = compute_shap_values(explainer, X_proc, model)

                    base_val = (
                        explainer.expected_value
                        if not isinstance(explainer.expected_value, (list, np.ndarray))
                        else (explainer.expected_value[1]
                              if hasattr(explainer.expected_value, "__len__") and
                              len(explainer.expected_value) > 1
                              else float(explainer.expected_value))
                    )

                    exp = shap.Explanation(
                        values=sv[0],
                        base_values=float(base_val),
                        data=X_proc[0],
                        feature_names=FEATURE_NAMES,
                    )

                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(exp, show=False, max_display=12)
                    st.pyplot(plt.gcf(), clear_figure=True)

                except Exception as shap_err:
                    # Fallback: plain bar chart
                    sv_1d = sv[0] if sv.ndim > 1 else sv
                    order = np.argsort(np.abs(sv_1d))[-12:][::-1]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#d73027" if v > 0 else "#4575b4" for v in sv_1d[order]]
                    ax.barh(
                        [FEATURE_NAMES[i] for i in order],
                        sv_1d[order],
                        color=colors,
                    )
                    ax.axvline(0, color="black", lw=0.8)
                    ax.set_xlabel("SHAP value (impact on approval probability)")
                    ax.set_title("Feature contributions")
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)

            st.caption(
                "🔵 Blue bars push towards rejection · 🔴 Red bars push towards approval · "
                "Bar length = magnitude of impact."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)

    # ── Fairness notice ────────────────────────────────────────────────────────
    with st.expander("ℹ️ Fairness & ethical considerations"):
        st.markdown(
            """
            **Known limitations**
            - Trained on 1,000 samples from 1970s West Germany — not representative of modern populations.
            - The model was audited for gender and age group gaps; gaps remain after mitigation.
            - Features like *foreign worker* and *employment history* can encode socioeconomic disadvantage.

            **Mitigation applied**
            - ExponentiatedGradient with Demographic Parity constraint (fairlearn) was applied during
              training; it reduced but did not eliminate demographic parity gaps.

            **What still needs work**
            - Causal analysis of features that proxy for protected attributes.
            - Ongoing monitoring for distribution shift.
            - Human review of every marginal (near-0.5 probability) decision.

            See `reports/evaluation_report.md` for full fairness audit results.
            """
        )
