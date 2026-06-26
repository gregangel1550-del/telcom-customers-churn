from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "xgb_churn_model.pkl"


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]
CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_bin",
]
TARGET_COL = "Churn"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def make_tenure_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure" not in df.columns:
        raise ValueError("Missing required column: tenure")

    max_tenure = max(float(df["tenure"].max()), 72.0)
    bins = [0, 12, 36, max_tenure]
    labels = ["new", "mid", "loyal"]

    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(str)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_telco_data(df)
    df = make_tenure_bin(df)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def get_model_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_features(df)

    expected = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")

    return df[expected].copy()


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    return joblib.load(path)


def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def score_dataframe(df: pd.DataFrame, preprocessor: Any, model: Any) -> pd.DataFrame:
    original = df.copy()
    X = get_model_input_frame(df)
    X_processed = preprocessor.transform(X)

    probs = model.predict_proba(X_processed)[:, 1]
    preds = (probs >= 0.4).astype(int)

    scored = original.copy()
    scored["churn_probability"] = probs
    scored["predicted_churn"] = preds
    scored["risk_band"] = pd.cut(
        scored["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return scored


def build_single_customer_input(payload: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    df = prepare_features(df)

    total_customers = len(df)
    churn_rate = float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else np.nan
    avg_tenure = float(df["tenure"].mean()) if "tenure" in df.columns else np.nan
    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else np.nan

    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "avg_tenure": avg_tenure,
        "avg_monthly_charges": avg_monthly,
    }


def get_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    out = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return out


def get_feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + BINARY_FEATURES + cat_names
