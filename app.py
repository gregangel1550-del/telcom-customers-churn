"""
Customer Churn Prediction — Streamlit App (Local)
==================================================
Run with:  streamlit run app.py
Opens at:  http://localhost:8501

No Docker, no cloud, no deployment needed.
Just Python + your two .pkl model files.

File structure required:
  your-folder/
  ├── app.py                  ← this file
  ├── requirements.txt
  ├── .streamlit/
  │   └── config.toml
  └── models/
      ├── preprocessor.pkl
      └── xgb_churn_model.pkl
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the VERY FIRST st.* call in the script
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Prediction Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent
THRESHOLD = 0.4   # decision threshold defined in Phase 1

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — injected once at startup
# ─────────────────────────────────────────────────────────────────────────────
# WHY inject CSS:
# Streamlit's default styling is functional but generic.
# A few targeted CSS rules make the app look polished and professional
# without needing a separate frontend framework.

st.markdown("""
<style>
/* ── Make metric labels slightly smaller and muted ── */
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    color: #6B7280 !important;
}

/* ── Metric value: larger and bold ── */
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Tab font size ── */
button[data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* ── Primary button styling ── */
.stButton > button[kind="primary"] {
    background-color: #1D9E75 !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px !important;
    border-radius: 8px !important;
}

/* ── Sidebar title ── */
[data-testid="stSidebar"] h2 {
    color: #1D9E75;
}

/* ── Expander header ── */
details summary {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — cached so disk reads happen only ONCE per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_models():
    """
    Load preprocessor and XGBoost model from the models/ folder.

    @st.cache_resource:
      Streamlit reruns this entire script on every user interaction.
      Without caching, joblib.load() would read the .pkl files from disk
      hundreds of times per session — adding ~500ms per click.
      cache_resource runs this function once, keeps the result in RAM,
      and returns the cached objects on every subsequent rerun.

    Returns (preprocessor, model) or (None, None) if files are missing.
    """
    preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"
    model_path        = BASE_DIR / "models" / "xgb_churn_model.pkl"

    if not preprocessor_path.exists():
        return None, None, f"Missing: {preprocessor_path}"
    if not model_path.exists():
        return None, None, f"Missing: {model_path}"

    preprocessor = joblib.load(preprocessor_path)
    model        = joblib.load(model_path)
    return preprocessor, model, None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — must match Phase 3 exactly
# ─────────────────────────────────────────────────────────────────────────────

def make_tenure_bin(tenure: int) -> str:
    """
    Reproduce the tenure_bin feature from Phase 3 notebook exactly.
    The preprocessor.pkl was fitted AFTER this column was created,
    so it must exist in every DataFrame we send to preprocessor.transform().

    Bins:
      new   → 0–12 months   (47.7% churn rate in EDA)
      mid   → 13–36 months  (26.1% churn rate)
      loyal → 37+ months    (6.2% churn rate)
    """
    if tenure <= 12:
        return "new"
    elif tenure <= 36:
        return "mid"
    else:
        return "loyal"


def get_feature_names(preprocessor) -> list:
    """
    Rebuild the full ordered list of feature names after preprocessing.
    Required for SHAP axis labels — without this, SHAP plots show
    column indices (0, 1, 2...) instead of readable names.
    """
    numeric_features     = ["tenure", "MonthlyCharges", "TotalCharges"]
    binary_features      = ["gender", "SeniorCitizen", "Partner", "Dependents",
                             "PhoneService", "PaperlessBilling"]
    categorical_features = ["MultipleLines", "InternetService", "OnlineSecurity",
                             "OnlineBackup", "DeviceProtection", "TechSupport",
                             "StreamingTV", "StreamingMovies", "Contract",
                             "PaymentMethod", "tenure_bin"]
    try:
        ohe_names = (preprocessor
                     .named_transformers_["cat"]["encoder"]
                     .get_feature_names_out(categorical_features))
        return numeric_features + binary_features + list(ohe_names)
    except Exception:
        return numeric_features + binary_features + categorical_features


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(customer_dict: dict, preprocessor, model) -> dict:
    """
    End-to-end prediction pipeline for a single customer.

    Steps:
      1. dict → DataFrame  (preprocessor.transform expects a DataFrame)
      2. Add tenure_bin    (feature engineering from Phase 3)
      3. preprocessor.transform()  (scale + encode using training stats)
      4. model.predict_proba()     (returns probability of churn)
      5. Derive risk tier, recommendation, SHAP values

    Why steps 1-3 must happen in this order:
      - The preprocessor was fitted on a DataFrame that already contained
        tenure_bin. Sending a DataFrame without it causes a KeyError.
      - We use .transform() not .fit_transform() — the scaler uses the
        mean/std it learned from training data, NOT from this customer's
        data. Using fit_transform here would be data leakage.
    """

    # Step 1 & 2
    df = pd.DataFrame([customer_dict])
    df["tenure_bin"] = df["tenure"].apply(make_tenure_bin)

    # Step 3
    X = preprocessor.transform(df)

    # Step 4 — predict_proba returns [[prob_class0, prob_class1]]
    # We take [:, 1] = probability of churn (class 1)
    probability = float(model.predict_proba(X)[0, 1])
    flagged     = probability >= THRESHOLD

    # Step 5a — risk tier
    if probability < 0.35:
        risk, badge_color = "Low",    "#27AE60"
    elif probability < 0.60:
        risk, badge_color = "Medium", "#E67E22"
    else:
        risk, badge_color = "High",   "#C0392B"

    # Step 5b — retention recommendation
    recs = {
        "Low": (
            "No immediate action required. Schedule a standard check-in at the "
            "next billing cycle. Monitor for sudden usage changes."
        ),
        "Medium": (
            "Proactive outreach recommended at the next contact opportunity. "
            "Offer a loyalty incentive or free add-on trial if the customer "
            "raises any concerns about value or pricing."
        ),
        "High": (
            "PRIORITY — Immediate outreach required within 48 hours. "
            "Escalate to a senior retention specialist. Lead with a concrete "
            "offer: contract upgrade discount, free TechSupport trial, or "
            "OnlineSecurity bundle to close the perceived value gap."
        )
    }

    # Step 5c — SHAP values for explainability
    # TreeExplainer uses XGBoost's tree structure for exact (non-approximate) SHAP.
    # Much faster than KernelExplainer for tree models.
    shap_df = None
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feat_names  = get_feature_names(preprocessor)
        shap_df = (
            pd.DataFrame({"feature": feat_names, "shap_value": shap_values[0]})
            .sort_values("shap_value", key=abs, ascending=False)
            .head(12)
            .reset_index(drop=True)
        )
    except Exception:
        pass   # SHAP unavailable — app still works, just no explanation chart

    return {
        "probability":      probability,
        "flagged":          flagged,
        "risk":             risk,
        "badge_color":      badge_color,
        "recommendation":   recs[risk],
        "shap_df":          shap_df,
        "tenure_bin":       df["tenure_bin"].iloc[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def gauge_chart(probability: float, color: str) -> go.Figure:
    """
    Plotly gauge showing churn probability.

    Design decisions:
    - Three coloured zones (green/amber/red) give instant visual context
    - Threshold line at 40% shows the exact decision boundary
    - delta shows distance from threshold so the user understands
      how borderline or extreme the case is
    """
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(probability * 100, 1),
        number = {"suffix": "%", "font": {"size": 42, "color": color}},
        delta  = {
            "reference": THRESHOLD * 100,
            "suffix": "% vs threshold",
            "font": {"size": 13}
        },
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#BDC3C7"},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35],  "color": "#EAFAF1"},
                {"range": [35, 60],  "color": "#FEF9E7"},
                {"range": [60, 100], "color": "#FDEDEC"},
            ],
            "threshold": {
                "line":      {"color": "#2C3E50", "width": 3},
                "thickness": 0.8,
                "value":     THRESHOLD * 100
            }
        },
        title = {"text": "Churn Probability", "font": {"size": 16, "color": "#5D6D7E"}}
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=70, b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def shap_chart(shap_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of SHAP values for the top 12 features.

    Red = feature pushed prediction TOWARD churn (positive SHAP)
    Green = feature pushed prediction AWAY from churn (negative SHAP)

    The bars are sorted by absolute value so the most impactful
    feature is always at the top — easy to scan for the retention team.
    """
    colors = ["#E74C3C" if v > 0 else "#27AE60" for v in shap_df["shap_value"]]

    # Clean OneHotEncoded feature names for display
    labels = (shap_df["feature"]
              .str.replace("_", " ", regex=False)
              .str.replace("No internet service", "(no internet)", regex=False)
              .str.replace("No phone service",    "(no phone)",    regex=False))

    fig = go.Figure(go.Bar(
        x           = shap_df["shap_value"],
        y           = labels,
        orientation = "h",
        marker_color = colors,
        marker_line_width = 0,
        text        = [f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition = "outside",
        textfont     = {"size": 11}
    ))
    fig.update_layout(
        title      = {"text": "Top drivers for this prediction (SHAP)", "font": {"size": 14}},
        xaxis_title = "← reduces churn risk  |  increases churn risk →",
        xaxis       = {"gridcolor": "#F2F3F4", "zeroline": True,
                       "zerolinecolor": "#2C3E50", "zerolinewidth": 1.5},
        yaxis       = {"autorange": "reversed"},
        height      = 430,
        margin      = dict(t=50, b=30, l=10, r=80),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
    return fig


def risk_badge(risk: str, color: str):
    """HTML badge for the risk level — shown prominently next to the gauge."""
    icons = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}
    st.markdown(f"""
        <div style="
            background: {color};
            color: white;
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 2px;
            margin-bottom: 12px;
        ">
            {icons[risk]}&nbsp; {risk.upper()} RISK
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    """
    Persistent sidebar — visible at all times regardless of scroll.
    Contains: branding, how-to-use, model stats, risk zone guide.
    """
    with st.sidebar:
        st.markdown("## 📊 Churn Predictor")
        st.caption("Customer Retention Intelligence Tool")
        st.divider()

        st.markdown("#### How to use")
        st.markdown("""
1. Fill in the customer's details in the **Single Customer** tab
2. Click **Run Prediction**
3. Read the probability gauge and SHAP explanation
4. Act on the recommendation
        """)

        st.divider()
        st.markdown("#### Model Performance")

        c1, c2 = st.columns(2)
        c1.metric("AUC-ROC",   "0.843")
        c2.metric("Recall",    "72%")
        c1.metric("Precision", "61%")
        c2.metric("Threshold", "40%")

        st.divider()
        st.markdown("#### Risk Zones")
        st.markdown("""
🟢 **Low** — below 35%  
🟡 **Medium** — 35 – 60%  
🔴 **High** — above 60%

The **decision threshold is 40%** — customers above this are flagged for outreach.
        """)

        st.divider()
        st.markdown("#### About this model")
        st.markdown("""
**Algorithm:** XGBoost  
**Dataset:** IBM Telco Churn (7,043 customers)  
**Explainability:** SHAP TreeExplainer  
**Imbalance handling:** scale_pos_weight = 2.77
        """)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CUSTOMER TAB — input form
# ─────────────────────────────────────────────────────────────────────────────

def render_input_form() -> dict:
    """
    Customer feature input form.

    Layout decisions:
    - 3-column grid keeps form compact, no excessive scrolling
    - Sliders for numeric fields: give real-time visual feedback
    - tenure_bin shown live so user understands lifecycle grouping
    - Add-ons in a separate row (important for SHAP-driven insights)
    - Demographics in a collapsed expander — lower SHAP importance,
      so they don't dominate the visual hierarchy of the form
    """

    st.markdown("### Customer Details")

    # ── Section 1: Contract & Billing ────────────────────────────────────────
    st.markdown("**Contract & Billing**")
    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 6,
                            help="Months the customer has been with the company")
        tb = make_tenure_bin(tenure)
        icons = {"new": "🔴", "mid": "🟡", "loyal": "🟢"}
        st.caption(f"Lifecycle stage: {icons[tb]} **{tb.capitalize()}**"
                   f"  ({['highest', 'moderate', 'lowest'][['new','mid','loyal'].index(tb)]} churn risk group)")

    with c2:
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0, 0.5,
                             help="Current monthly bill in USD")

    with c3:
        default_total = round(tenure * monthly, 2)
        total = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                 float(default_total), 10.0,
                                 help="Cumulative charges to date. Auto-filled as tenure × monthly.")

    c4, c5, c6 = st.columns(3)
    with c4:
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"],
                                 help="Month-to-month customers churn at 42% vs 3% for 2-year")
    with c5:
        payment = st.selectbox("Payment Method",
                                ["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)", "Credit card (automatic)"])
    with c6:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    # ── Section 2: Add-on Services ────────────────────────────────────────────
    st.markdown("**Add-on Services**")
    a1, a2, a3, a4 = st.columns(4)
    opts_internet = ["No", "Yes", "No internet service"]
    opts_phone    = ["No", "Yes", "No phone service"]

    with a1:
        sec = st.selectbox("Online Security",    opts_internet)
    with a2:
        backup = st.selectbox("Online Backup",   opts_internet)
    with a3:
        support = st.selectbox("Tech Support",   opts_internet)
    with a4:
        device = st.selectbox("Device Protection", opts_internet)

    b1, b2 = st.columns(2)
    with b1:
        tv = st.selectbox("Streaming TV",     opts_internet)
    with b2:
        movies = st.selectbox("Streaming Movies", opts_internet)

    # ── Section 3: Demographics (collapsed) ───────────────────────────────────
    # Collapsed by default because demographic features have lower SHAP
    # importance than contract/billing features (confirmed in Phase 5).
    # They're still included because they affect the model — just not front-and-centre.
    with st.expander("Demographics & Account (click to expand)", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with d2:
            senior = st.selectbox("Senior Citizen", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
        with d3:
            partner = st.selectbox("Partner", ["No", "Yes"])
        with d4:
            deps = st.selectbox("Dependents", ["No", "Yes"])

        d5, d6, d7 = st.columns(3)
        with d5:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with d6:
            lines = st.selectbox("Multiple Lines", opts_phone)
        with d7:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    return {
        "tenure":           tenure,
        "MonthlyCharges":   monthly,
        "TotalCharges":     total,
        "Contract":         contract,
        "PaymentMethod":    payment,
        "InternetService":  internet,
        "OnlineSecurity":   sec,
        "OnlineBackup":     backup,
        "TechSupport":      support,
        "DeviceProtection": device,
        "StreamingTV":      tv,
        "StreamingMovies":  movies,
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       deps,
        "PhoneService":     phone,
        "MultipleLines":    lines,
        "PaperlessBilling": paperless,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_results(result: dict):
    """
    Display prediction results in three rows:
      Row 1 — Gauge  |  Risk badge + recommendation
      Row 2 — Four metric tiles
      Row 3 — SHAP chart  |  How-to-read explanation
    """
    st.divider()
    st.markdown("## 🎯 Prediction Results")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    col_g, col_r = st.columns([1, 1])

    with col_g:
        st.plotly_chart(
            gauge_chart(result["probability"], result["badge_color"]),
            use_container_width=True
        )

    with col_r:
        st.markdown("#### Risk Level")
        risk_badge(result["risk"], result["badge_color"])

        st.markdown("#### Retention Recommendation")
        if result["risk"] == "High":
            st.error(result["recommendation"])
        elif result["risk"] == "Medium":
            st.warning(result["recommendation"])
        else:
            st.success(result["recommendation"])

        # Verdict banner
        pct = result["probability"] * 100
        if result["flagged"]:
            st.error(
                f"⚠️ **Flagged for outreach** — {pct:.1f}% exceeds the "
                f"{THRESHOLD*100:.0f}% decision threshold"
            )
        else:
            st.success(
                f"✓ **Not flagged** — {pct:.1f}% is below the "
                f"{THRESHOLD*100:.0f}% decision threshold"
            )

    # ── Row 2: metric strip ───────────────────────────────────────────────────
    st.markdown("#### Prediction Details")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Churn Probability",
        f"{result['probability']*100:.1f}%"
    )
    m2.metric(
        "Risk Level",
        result["risk"]
    )
    m3.metric(
        "Decision",
        "FLAG 🚨" if result["flagged"] else "PASS ✅",
        delta="Above threshold" if result["flagged"] else "Below threshold",
        delta_color="inverse"
    )
    m4.metric(
        "Lifecycle Stage",
        result["tenure_bin"].capitalize()
    )

    # ── Row 3: SHAP chart ─────────────────────────────────────────────────────
    st.markdown("#### Why did the model make this prediction?")
    sh_col, exp_col = st.columns([3, 2])

    with sh_col:
        if result["shap_df"] is not None:
            st.plotly_chart(
                shap_chart(result["shap_df"]),
                use_container_width=True
            )
            st.caption(
                "🔴 Red = pushes **toward** churn  "
                "  🟢 Green = pushes **away** from churn"
            )
        else:
            st.info("SHAP explanation is not available for this prediction.")

    with exp_col:
        st.markdown("**How to read this chart**")
        st.markdown("""
Each bar is one feature. The length shows how strongly that feature
influenced **this specific customer's** churn score.

- **Positive value (red)** → the feature increased the model's
  churn probability for this customer
- **Negative value (green)** → the feature decreased it
- Features are sorted by impact — the most important driver is at the top

**Use the top driver to personalise outreach:**

If `Contract Month-to-month` is the biggest red bar → lead with a
contract upgrade offer.

If `tenure` is the top red bar → focus on early-stage value
demonstration and onboarding support.

If `MonthlyCharges` is red → discuss whether a plan adjustment
could improve the customer's perception of value.
        """)

        if result["shap_df"] is not None:
            top = result["shap_df"].iloc[0]
            direction = "increases" if top["shap_value"] > 0 else "reduces"
            st.markdown("---")
            st.markdown(
                f"**Top driver for this customer:**  \n"
                f"`{top['feature'].replace('_', ' ')}` "
                f"{direction} churn risk "
                f"(SHAP = `{top['shap_value']:+.3f}`)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCORING TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_tab(preprocessor, model):
    """
    Upload a CSV → score all customers → download results.

    Why this tab exists:
    In production a retention team scores hundreds of customers every week
    from a CRM export. Entering them one-by-one via the form is impossible.
    Batch scoring is the realistic workflow.
    """
    st.markdown("### Batch Customer Scoring")
    st.markdown(
        "Upload a CSV with multiple customers. "
        "All rows are scored at once and you can download the results."
    )

    with st.expander("Required CSV columns"):
        st.code(
            "tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, "
            "InternetService, OnlineSecurity, OnlineBackup, TechSupport, "
            "DeviceProtection, StreamingTV, StreamingMovies, gender, "
            "SeniorCitizen, Partner, Dependents, PhoneService, "
            "MultipleLines, PaperlessBilling"
        )
        st.markdown(
            "Column names must match exactly (case-sensitive). "
            "Values must match the IBM Telco dataset format — "
            "e.g. `Yes`/`No` for binary fields, "
            "`Month-to-month` / `One year` / `Two year` for Contract."
        )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"File loaded: **{len(df):,} customers**")

            # Preview
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("Score All Customers", type="primary"):
                with st.spinner(f"Scoring {len(df):,} customers..."):

                    # Preprocessing — same steps as single prediction
                    df["TotalCharges"] = pd.to_numeric(
                        df["TotalCharges"], errors="coerce"
                    ).fillna(0)
                    df["tenure_bin"] = df["tenure"].apply(make_tenure_bin)

                    X    = preprocessor.transform(df)
                    prob = model.predict_proba(X)[:, 1]

                    df["churn_probability_%"] = (prob * 100).round(1)
                    df["risk_level"] = pd.cut(
                        prob,
                        bins=[0, 0.35, 0.60, 1.0],
                        labels=["Low", "Medium", "High"]
                    )
                    df["flagged_for_outreach"] = (prob >= THRESHOLD).astype(int)
                    df = df.sort_values("churn_probability_%", ascending=False)

                st.success("✅ Scoring complete!")

                # Summary strip
                total   = len(df)
                flagged = df["flagged_for_outreach"].sum()
                high    = (df["risk_level"] == "High").sum()
                avg_p   = df["churn_probability_%"].mean()

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total customers",      f"{total:,}")
                s2.metric("Flagged for outreach", f"{flagged:,}")
                s3.metric("High risk",            f"{high:,}")
                s4.metric("Average probability",  f"{avg_p:.1f}%")

                # Distribution chart
                fig = px.histogram(
                    df, x="churn_probability_%",
                    nbins=30,
                    color="risk_level",
                    color_discrete_map={
                        "Low":    "#27AE60",
                        "Medium": "#E67E22",
                        "High":   "#C0392B"
                    },
                    title="Churn probability distribution across all customers",
                    labels={"churn_probability_%": "Churn Probability (%)"},
                    category_orders={"risk_level": ["Low", "Medium", "High"]}
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top 20 highest-risk customers
                st.markdown("#### Top 20 highest-risk customers")
                display_cols = [c for c in [
                    "churn_probability_%", "risk_level", "flagged_for_outreach",
                    "tenure", "MonthlyCharges", "Contract", "tenure_bin"
                ] if c in df.columns]
                st.dataframe(df[display_cols].head(20), use_container_width=True)

                # Download button
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download full scored CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.markdown(
                "Make sure your CSV columns match the required names exactly. "
                "Column names are **case-sensitive**."
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    App entry point. Streamlit calls this on every rerun.

    Why st.session_state for the prediction result:
    Streamlit reruns the full script on every interaction (slider, dropdown...).
    Without session_state, the prediction result disappears the moment the
    user touches any input. Storing the result in session_state keeps it
    visible across reruns until a new prediction is explicitly triggered.
    """

    render_sidebar()

    # App header
    st.title("📊 Customer Churn Prediction")
    st.markdown(
        "Identify customers at risk of cancelling their subscription so the "
        "retention team can intervene **before** they leave."
    )

    # Load models — runs once, cached permanently
    preprocessor, model, error = load_models()

    # ── Model files not found ──────────────────────────────────────────────
    if error:
        st.error(f"⚠️ {error}")
        st.markdown("""
**Setup instructions:**

1. Run your project notebooks in order:
   ```
   02_feature_engineering.ipynb  →  generates  models/preprocessor.pkl
   03_modeling.ipynb             →  generates  models/xgb_churn_model.pkl
   ```
2. Make sure both `.pkl` files are in the `models/` folder next to `app.py`
3. Restart the app: `streamlit run app.py`
        """)
        st.stop()   # halt execution — don't render the rest of the app

    # ── Two tabs ──────────────────────────────────────────────────────────
    tab_single, tab_batch = st.tabs(["🔍  Single Customer", "📁  Batch Scoring"])

    # ── Tab 1: single prediction ──────────────────────────────────────────
    with tab_single:
        customer = render_input_form()

        st.markdown("")
        clicked = st.button(
            "🚀  Run Prediction",
            type="primary",
            use_container_width=True
        )

        if clicked:
            with st.spinner("Running prediction and computing SHAP values..."):
                result = run_prediction(customer, preprocessor, model)
            st.session_state["result"]   = result
            st.session_state["customer"] = customer

        # Show result if one exists in session (persists across reruns)
        if "result" in st.session_state:
            render_results(st.session_state["result"])

    # ── Tab 2: batch scoring ──────────────────────────────────────────────
    with tab_batch:
        render_batch_tab(preprocessor, model)


if __name__ == "__main__":
    main()
