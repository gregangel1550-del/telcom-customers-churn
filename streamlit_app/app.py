from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    RAW_DATA_PATH,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_single_customer_input,
    clean_telco_data,
    get_feature_importance,
    get_feature_names_from_preprocessor,
    load_model,
    load_preprocessor,
    load_raw_data,
    prepare_features,
    score_dataframe,
    summarize_dataset,
)


st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def cached_load_data(path_str: str) -> pd.DataFrame:
    return load_raw_data(Path(path_str))


@st.cache_resource
def cached_load_preprocessor():
    return load_preprocessor()


@st.cache_resource
def cached_load_model():
    return load_model()


def render_header() -> None:
    st.title("Telco Customer Churn Dashboard")
    st.caption(
        "Interactive dashboard, batch scoring, and single-customer churn prediction."
    )


def render_sidebar() -> str:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "EDA Dashboard",
            "Single Prediction",
            "Batch Prediction",
            "Model Insights",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Expected model artifacts:\n"
        "- models/preprocessor.pkl\n"
        "- models/xgb_churn_model.pkl"
    )
    return page


def metric_row(df: pd.DataFrame) -> None:
    summary = summarize_dataset(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{summary['total_customers']:,}")
    c2.metric("Churn Rate", f"{summary['churn_rate'] * 100:.1f}%")
    c3.metric("Avg Tenure", f"{summary['avg_tenure']:.1f} months")
    c4.metric("Avg Monthly Charge", f"${summary['avg_monthly_charges']:.2f}")


def overview_page(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    metric_row(df)

    clean_df = prepare_features(df)

    col1, col2 = st.columns(2)

    with col1:
        if "Churn" in clean_df.columns:
            churn_counts = (
                clean_df["Churn"]
                .map({0: "No Churn", 1: "Churn"})
                .value_counts()
                .reset_index()
            )
            churn_counts.columns = ["label", "count"]
            fig = px.bar(
                churn_counts,
                x="label",
                y="count",
                color="label",
                title="Target Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract = clean_df["Contract"].value_counts().reset_index()
        contract.columns = ["Contract", "count"]
        fig = px.pie(
            contract,
            names="Contract",
            values="count",
            title="Contract Mix",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)


def eda_page(df: pd.DataFrame) -> None:
    st.subheader("EDA Dashboard")
    clean_df = prepare_features(df)

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        fig = px.histogram(
            clean_df,
            x="tenure",
            nbins=30,
            color="Churn" if "Churn" in clean_df.columns else None,
            title="Tenure Distribution",
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        fig = px.histogram(
            clean_df,
            x="MonthlyCharges",
            nbins=30,
            color="Churn" if "Churn" in clean_df.columns else None,
            title="Monthly Charges Distribution",
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    with row2_col1:
        churn_by_contract = (
            clean_df.groupby("Contract", dropna=False)["Churn"]
            .mean()
            .reset_index()
            .sort_values("Churn", ascending=False)
        )
        fig = px.bar(
            churn_by_contract,
            x="Contract",
            y="Churn",
            title="Churn Rate by Contract",
        )
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        churn_by_internet = (
            clean_df.groupby("InternetService", dropna=False)["Churn"]
            .mean()
            .reset_index()
            .sort_values("Churn", ascending=False)
        )
        fig = px.bar(
            churn_by_internet,
            x="InternetService",
            y="Churn",
            title="Churn Rate by Internet Service",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Churn by Payment Method")
    churn_by_payment = (
        clean_df.groupby("PaymentMethod", dropna=False)["Churn"]
        .mean()
        .reset_index()
        .sort_values("Churn", ascending=False)
    )
    fig = px.bar(
        churn_by_payment,
        x="PaymentMethod",
        y="Churn",
        title="Churn Rate by Payment Method",
    )
    st.plotly_chart(fig, use_container_width=True)


def single_prediction_page(preprocessor, model) -> None:
    st.subheader("Single Customer Prediction")
    st.write("Enter one customer profile and estimate churn probability.")

    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )

        with c3:
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c5:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c6:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=0.1,
            )
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=1.0,
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = build_single_customer_input(payload)
        scored = score_dataframe(input_df, preprocessor, model)

        prob = float(scored["churn_probability"].iloc[0])
        pred = int(scored["predicted_churn"].iloc[0])
        band = scored["risk_band"].iloc[0]

        st.markdown("### Prediction Result")
        a, b, c = st.columns(3)
        a.metric("Churn Probability", f"{prob:.1%}")
        b.metric("Predicted Class", "Churn" if pred == 1 else "No Churn")
        c.metric("Risk Band", str(band))

        st.progress(min(max(prob, 0.0), 1.0))

        st.dataframe(scored, use_container_width=True)


def batch_prediction_page(preprocessor, model) -> None:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV with the same raw columns as the training dataset.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data")
        st.dataframe(batch_df.head(), use_container_width=True)

        try:
            scored = score_dataframe(batch_df, preprocessor, model)

            st.success("Batch scoring completed.")
            st.dataframe(scored.head(20), use_container_width=True)

            csv_buffer = io.StringIO()
            scored.to_csv(csv_buffer, index=False)

            st.download_button(
                label="Download scored CSV",
                data=csv_buffer.getvalue(),
                file_name="batch_churn_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")


def model_insights_page(preprocessor, model, df: pd.DataFrame) -> None:
    st.subheader("Model Insights")

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    importance_df = get_feature_importance(model, feature_names)

    if importance_df.empty:
        st.warning("This model does not expose feature_importances_.")
        return

    top_n = st.slider("Top features", 5, 30, 15)
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importances",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance_df.head(30), use_container_width=True)

    st.markdown("### Optional SHAP")
    st.caption(
        "If SHAP is installed and compatible with your XGBoost version, you can add "
        "a SHAP summary plot here. The current app already provides built-in feature importance."
    )

    clean_df = clean_telco_data(df)
    if "Churn" in clean_df.columns:
        churn_by_tenure_bin = (
            prepare_features(df)
            .groupby("tenure_bin")["Churn"]
            .mean()
            .reset_index()
        )
        fig2 = px.bar(
            churn_by_tenure_bin,
            x="tenure_bin",
            y="Churn",
            title="Observed Churn Rate by tenure_bin",
        )
        st.plotly_chart(fig2, use_container_width=True)


def main() -> None:
    render_header()
    page = render_sidebar()

    try:
        raw_df = cached_load_data(str(RAW_DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    try:
        preprocessor = cached_load_preprocessor()
        model = cached_load_model()
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    if page == "Overview":
        overview_page(raw_df)
    elif page == "EDA Dashboard":
        eda_page(raw_df)
    elif page == "Single Prediction":
        single_prediction_page(preprocessor, model)
    elif page == "Batch Prediction":
        batch_prediction_page(preprocessor, model)
    elif page == "Model Insights":
        model_insights_page(preprocessor, model, raw_df)


if __name__ == "__main__":
    main()
