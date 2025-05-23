from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    age: float
    income_range: str
    education: str

@app.post("/predict")
def predict(data: InputData):
    # Define categories
    income_categories = ["Low", "Medium", "High", "Very High"]
    education_categories = ["High School", "Other", "Graduate", "Post Graduate"]

    # One-hot encode with drop='first'
    income_encoded = [1 if data.income_range.lower() == cat.lower() else 0 for cat in income_categories[1:]]
    education_encoded = [1 if data.education.lower() == cat.lower() else 0 for cat in education_categories[1:]]

    # Combine features
    input_data = np.array([income_encoded + education_encoded + [data.age]])

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]
    return {"renewal_probability": round(probability, 2)}
