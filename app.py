from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

# Load model and encoder
with open("churn_rf_healthy_meals.pkl", "rb") as f:
    model = pickle.load(f)
with open("churn_encoder_healthy_meals.pkl", "rb") as f:
    encoder = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    age: float
    income_level: str
    education: str
    device_type: str
    tech_comfort_score: int

@app.post("/predict")
def predict(data: InputData):
    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        'INCOME_LEVEL': data.income_level,
        'EDUCATION':    data.education,
        'DEVICE_TYPE':  data.device_type,
    }])

    # Apply saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features first, then encoded dummies — must match training column order
    numeric_df = pd.DataFrame([{
        'AGE':                data.age,
        'TECH_COMFORT_SCORE': data.tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    return {
        "renewal_probability": round(float(probability), 2),
        "churn_risk": risk
    }
