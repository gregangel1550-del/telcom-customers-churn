from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from schema import CustomerFeatures, PredictionResponse
from predict import predict_churn

# ── Application setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
Predicts the probability that a telecom customer will churn within 30 days.

## How to use

Send a POST request to `/predict` with the customer's current features.
The API returns:
- A churn probability (0.0 to 1.0)
- A risk label (Low / Medium / High)
- A plain English recommendation for the retention team

## Model

XGBoost classifier trained on IBM Telco Customer Churn dataset.
Primary metric: AUC-ROC = 0.843 on held-out test set.
Decision threshold: 0.4 (optimised for recall).
    """,
    version="1.0.0",
    contact={
        "name": "Gregoriant Angelo Bere",
        "url": "https://github.com/yourusername/customer-churn-prediction"
    }
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the API to be called from a browser (e.g. a frontend dashboard)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "online",
            "api": "Customer Churn Prediction API",
            "version": "1.0.0",
            "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check for monitoring systems."""
    return {"status": "healthy",
            "model_loaded": True,
            "timestamp": time.time()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Accepts a JSON body with the customer's current features.
    Returns a churn probability, risk tier, and retention recommendation.
    """
    try:
        # Convert Pydantic model to plain dict for the prediction function
        customer_dict = customer.model_dump()

        # Run prediction
        result = predict_churn(customer_dict)

        return PredictionResponse(**result)

    except Exception as e:
        # Return a structured error instead of crashing with a 500
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    """
    Predict churn probability for a list of customers (max 100).

    Useful for scoring all customers in a CRM export at once.
    """
    if len(customers) > 100:
        raise HTTPException(status_code=400,
                            detail="Batch size limited to 100 customers per request.")

    results = []
    for customer in customers:
        try:
            customer_dict = customer.model_dump()
            result = predict_churn(customer_dict)
            results.append({"status": "success", **result})
        except Exception as e:
            results.append({"status": "error", "detail": str(e)})

    return {"predictions": results, "count": len(results)}