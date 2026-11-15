
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Purchase Intent Analyzer",
    page_icon="🛍️",
    layout="wide"
)


# =====================================================
# Custom White Theme - Improved Text Contrast
# =====================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #FFFFFF;
    color: #263445;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main Heading */
h1 {
    color: #17365D !important;
    font-size: 36px !important;
    font-weight: 750 !important;
}

/* Section Headings */
h2 {
    color: #1D4E78 !important;
    font-size: 25px !important;
    font-weight: 700 !important;
    margin-top: 25px !important;
}

h3 {
    color: #285B85 !important;
    font-weight: 650 !important;
}

/* General Text */
p {
    color: #34465A !important;
}

label {
    color: #263E56 !important;
    font-weight: 500 !important;
}

/* General Streamlit Text */
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #475569 !important;
}

/* Metrics Cards */
[data-testid="stMetric"] {
    background-color: #F3F8FC;
    border: 1px solid #DCE8F2;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #38546F !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-weight: 750 !important;
}

[data-testid="stMetricDelta"] {
    color: #287A62 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #F7FAFD;
    border: 1.5px dashed #8AAFCB;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #294761 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #245B85 !important;
    color: #FFFFFF !important;
    border: 1px solid #245B85 !important;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button p,
.stDownloadButton > button p,
.stButton > button span,
.stDownloadButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #17365D !important;
    color: #FFFFFF !important;
    border-color: #17365D !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE8F2;
    border-radius: 10px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stAlert"] p {
    color: #34465A !important;
}

/* Divider */
hr {
    border-color: #DCE8F2 !important;
}

/* Captions */
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F5F9FC;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #34465A !important;
}

/* Chart Containers */
[data-testid="stVegaLiteChart"] {
    background-color: #FFFFFF;
    border: 1px solid #E2EAF1;
    border-radius: 12px;
    padding: 10px;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    color: #263E56 !important;
    background-color: #FFFFFF !important;
}

/* Download Button Icon */
.stDownloadButton svg,
.stButton svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# Model Configuration
# =====================================================

MODEL_PATH = Path(__file__).parent / "artifacts" / "purchase_intent_pipeline.joblib"


# =====================================================
# Header
# =====================================================

st.title("🛍️ Will This Visitor Buy?")

st.write(
    "Upload a CSV containing browsing-session features. The saved model will estimate "
    "purchase probability for each session. **Uploading a file does not train or retrain the model.**"
)

st.markdown("---")


# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Run all cells in purchase_intent_training.ipynb "
        "to create artifacts/purchase_intent_pipeline.joblib, then restart the app."
    )
    st.stop()


artifact = load_artifact(MODEL_PATH)

model = artifact["pipeline"]
feature_columns = artifact["feature_columns"]

st.caption(
    f"Loaded saved model: {artifact.get('model_name', 'Trained classifier')}"
)


# =====================================================
# Upload CSV
# =====================================================

uploaded_file = st.file_uploader(
    "📂 Upload session data (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        incoming = pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()


    # =================================================
    # 1. File Preview
    # =================================================

    st.subheader("1. File Preview")

    st.write(
        f"Rows: **{len(incoming):,}** · "
        f"Columns: **{len(incoming.columns)}**"
    )

    st.dataframe(
        incoming.head(10),
        use_container_width=True
    )


    # Revenue is an optional label for retrospective evaluation only;
    # never passed to model.predict.

    has_actual = "Revenue" in incoming.columns

    X_new = incoming.drop(
        columns=["Revenue"],
        errors="ignore"
    ).copy()


    # =================================================
    # Validate Features
    # =================================================

    missing = [
        c for c in feature_columns
        if c not in X_new.columns
    ]

    extra = [
        c for c in X_new.columns
        if c not in feature_columns
    ]


    if missing:

        st.error(
            "The uploaded CSV is missing required feature columns: "
            + ", ".join(missing)
        )

        st.info(
            "Use the sample CSV generated by the training notebook as a template."
        )

        st.stop()


    if extra:

        st.warning(
            "Extra columns will be ignored: "
            + ", ".join(extra)
        )


    X_new = X_new[feature_columns]


    if X_new.empty:

        st.warning(
            "The uploaded file contains no session rows."
        )

        st.stop()


    # =================================================
    # Prediction
    # =================================================

    try:

        probabilities = model.predict_proba(X_new)[:, 1]

    except Exception as exc:

        st.error(
            "Prediction failed. Check that the CSV uses the same feature meanings and data "
            f"types as the training dataset. Details: {exc}"
        )

        st.stop()


    # =================================================
    # Business Recommendations
    # =================================================

    # Illustrative business policy;
    # thresholds are not trained model parameters.

    def suggested_action(p):

        if p >= 0.65:
            return "No action"

        elif p >= 0.30:
            return "Show chat assistant"

        return "Consider coupon"


    # =================================================
    # Create Results
    # =================================================

    results = incoming.copy()

    results["purchase_probability"] = probabilities

    results["prediction"] = np.where(
        probabilities >= 0.50,
        "Likely to purchase",
        "Unlikely to purchase"
    )

    results["suggested_action"] = [
        suggested_action(p)
        for p in probabilities
    ]


    # =================================================
    # 2. Prediction Summary
    # =================================================

    st.subheader("2. Prediction Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Sessions Analyzed",
        f"{len(results):,}"
    )

    c2.metric(
        "Likely to Purchase",
        f"{(probabilities >= 0.50).sum():,}"
    )

    c3.metric(
        "Average Purchase Probability",
        f"{probabilities.mean():.1%}"
    )


    # =================================================
    # 3. Suggested Actions
    # =================================================

    st.subheader("3. Suggested Actions")

    st.bar_chart(
        results["suggested_action"].value_counts(),
        color="#397BA8"
    )


    # =================================================
    # 4. Session-Level Results
    # =================================================

    st.subheader("4. Session-Level Results")

    st.dataframe(
        results.sort_values(
            "purchase_probability",
            ascending=False
        ),
        use_container_width=True
    )


    # =================================================
    # Download Predictions
    # =================================================

    csv_bytes = results.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_bytes,
        file_name="purchase_intent_predictions.csv",
        mime="text/csv"
    )


    # =================================================
    # 5. Retrospective Evaluation
    # =================================================

    if has_actual:

        st.info(
            "The uploaded file contains a Revenue column. It was excluded from prediction inputs. "
            "It can be used only for retrospective comparison because actual outcomes are already known."
        )

        actual = (
            incoming["Revenue"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "1": 1,
                "yes": 1,
                "false": 0,
                "0": 0,
                "no": 0
            })
        )

        valid = actual.notna()


        if valid.any():

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score
            )

            y_true = actual[valid].astype(int)

            y_prob = probabilities[valid.to_numpy()]

            y_pred = (y_prob >= 0.50).astype(int)


            st.subheader(
                "5. Retrospective Comparison "
                "(Only When Revenue Labels Are Present)"
            )


            m1, m2, m3, m4 = st.columns(4)


            m1.metric(
                "Accuracy",
                f"{accuracy_score(y_true, y_pred):.3f}"
            )

            m2.metric(
                "Precision",
                f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
            )

            m3.metric(
                "Recall",
                f"{recall_score(y_true, y_pred, zero_division=0):.3f}"
            )

            m4.metric(
                "F1",
                f"{f1_score(y_true, y_pred, zero_division=0):.3f}"
            )


            if y_true.nunique() == 2:

                st.metric(
                    "ROC-AUC",
                    f"{roc_auc_score(y_true, y_prob):.3f}"
                )


# =====================================================
# Footer
# =====================================================

st.divider()

st.caption(
    "Important: this app analyzes uploaded historical/session CSV data. It is not connected "
    "to a live store or tracking visitors automatically. Predictions are estimates, not guarantees. "
    "The suggested actions are illustrative business rules."
)
