from pydantic import BaseModel, Field
from typing import Literal


class CustomerFeatures(BaseModel):
    """
    Input schema — one customer's raw features.
    All fields mirror the original IBM Telco dataset columns.
    Pydantic automatically validates types and raises a clear error
    if the caller sends wrong data (e.g. a string where a number is expected).
    """

    # Numeric
    tenure: int = Field(..., ge=0, le=72,
                        description="Months the customer has been with the company")
    MonthlyCharges: float = Field(..., ge=0,
                                   description="Current monthly charge in USD")
    TotalCharges: float = Field(..., ge=0,
                                 description="Total charges to date in USD")

    # Binary (Yes/No)
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]

    # Categorical
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal["Electronic check",
                           "Mailed check",
                           "Bank transfer (automatic)",
                           "Credit card (automatic)"]

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 5,
                "MonthlyCharges": 85.5,
                "TotalCharges": 427.5,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "PaperlessBilling": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check"
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema — what the API returns for each customer.
    """
    churn_probability: float = Field(description="Probability the customer will churn (0.0 to 1.0)")
    churn_risk: Literal["Low", "Medium", "High"] = Field(description="Risk label derived from probability")
    recommendation: str = Field(description="Plain English action for the retention team")
    model_version: str = Field(description="Model identifier for audit trail")