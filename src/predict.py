from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("models/churn_model.joblib")


# --------------------------------------------------
# CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time customer churn prediction service",
    version="1.0.0"
)


# --------------------------------------------------
# DEFINE CUSTOMER INPUT
# --------------------------------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "customer-churn-model"
    }


# --------------------------------------------------
# REAL-TIME PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction = model.predict(customer_df)[0]

    probabilities = model.predict_proba(customer_df)[0]

    churn_probability = float(probabilities[1])

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percent": round(
            churn_probability * 100,
            2
        )
    }