# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from termcolor import colored

# Load model once
model = joblib.load("models/baseline_model.pkl")

app = FastAPI(title="Telco Churn Prediction API", version="1.0")

# Define input data format
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "🚀 Telco Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(pred),
        "churn_probability": round(prob * 100, 2)
    }

if __name__ == "__main__":
    print(colored("✅ API is live at: http://127.0.0.1:8000/docs", "green"))
