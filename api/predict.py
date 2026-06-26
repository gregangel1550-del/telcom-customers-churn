import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
MODEL_PATH        = BASE_DIR / "models" / "xgb_churn_model.pkl"

# ── Load artifacts once at startup, not on every request ─────────────────────
# Loading from disk on every API call would add ~500ms latency per request.
# Loading at module import time means the objects live in memory permanently.
print(f"Loading preprocessor from {PREPROCESSOR_PATH}")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print("Model and preprocessor loaded successfully.")

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.4
MODEL_VERSION = "xgboost-v1.0"


def make_tenure_bin(tenure: int) -> str:
    """
    Reproduce the exact tenure_bin feature engineering from Phase 3.
    Must be identical to what was done during training.
    """
    if tenure <= 12:
        return "new"
    elif tenure <= 36:
        return "mid"
    else:
        return "loyal"


def get_risk_label(probability: float) -> str:
    """Convert probability to a human-readable risk tier."""
    if probability < 0.35:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"


def get_recommendation(risk: str, probability: float) -> str:
    """Generate a plain English recommendation for the retention team."""
    recommendations = {
        "Low": (
            f"No immediate action required. "
            f"Churn probability: {probability:.0%}. "
            f"Standard engagement touchpoint recommended at next billing cycle."
        ),
        "Medium": (
            f"Monitor closely. Churn probability: {probability:.0%}. "
            f"Consider proactive outreach at next contact opportunity. "
            f"Offer loyalty incentive if engagement drops."
        ),
        "High": (
            f"PRIORITY — Immediate outreach recommended. "
            f"Churn probability: {probability:.0%}. "
            f"Escalate to retention specialist. "
            f"Offer contract upgrade discount or free add-on trial."
        )
    }
    return recommendations[risk]


def predict_churn(customer_data: dict) -> dict:
    """
    Main prediction function.

    Takes a dictionary of raw customer features (matching the original
    dataset schema), applies preprocessing, runs inference, and returns
    a structured prediction result.

    Parameters
    ----------
    customer_data : dict
        Raw customer features from the API request body.

    Returns
    -------
    dict
        churn_probability, churn_risk, recommendation, model_version
    """

    # Step 1: Convert dict to DataFrame
    # The preprocessor was fitted on a DataFrame — it expects the same format
    df = pd.DataFrame([customer_data])

    # Step 2: Reproduce feature engineering from Phase 3
    # This MUST match what was done during training exactly
    df["tenure_bin"] = df["tenure"].apply(make_tenure_bin)

    # Step 3: Apply the fitted preprocessor (scale, encode, impute)
    X_processed = preprocessor.transform(df)

    # Step 4: Run model inference
    probability = float(model.predict_proba(X_processed)[0, 1])

    # Step 5: Derive risk label and recommendation
    risk = get_risk_label(probability)
    recommendation = get_recommendation(risk, probability)

    return {"churn_probability": round(probability, 4),
            "churn_risk": risk,
            "recommendation": recommendation,
            "model_version": MODEL_VERSION}