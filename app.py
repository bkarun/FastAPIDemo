from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model and encoder from files uploaded to the Hugging Face Space
with open("churn_rf_healthy_meals.pkl", "rb") as f:
    model = pickle.load(f)

with open("churn_encoder_healthy_meals.pkl", "rb") as f:
    encoder = pickle.load(f)
    
app = FastAPI()

# Define input schema
class InputData(BaseModel):
    age: float
    income_level: str
    education: str
    device_type: str
    tech_comfort_score: int

@app.post("/predict")

def predict(data: InputData):
    age = data.age
    income_level = data.income_level
    education = data.education
    device_type = data.device_type
    tech_comfort_score = data.tech_comfort_score
    """
    Predict renewal probability for a single customer.
    The input values must be passed through the same encoder used during
    training. We build a one-row DataFrame with the raw categorical values
    (matching the column names the encoder was fit on), transform it, then
    combine with the numeric features in the same order as the training
    feature matrix:
        Training column order (from Step 4 of the training notebook):
        [AGE, TECH_COMFORT_SCORE, <encoded dummies in encoder order>]
    Bug note: do NOT recreate the one-hot logic by hand — case mismatches
    or category order differences will produce a constant all-zeros input
    and a constant prediction regardless of what the user enters.
    """

    # Build a single-row DataFrame with the raw categorical values.
    # Column names must match exactly what the encoder was fit on (UPPERCASE).
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform here)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Build the numeric part of the feature vector
    numeric_df = pd.DataFrame([{
        'AGE':               age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
    }])

    # Combine in the same column order as the training feature matrix:
    # numeric columns first, then encoded dummies
    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Predict: column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]

    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"
    return f"Renewal Probability: {probability:.2f}  |  Churn Risk: {risk}"
