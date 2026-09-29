from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Disease Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5174", "http://localhost:5174"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Load model and symptom list
model = joblib.load("models/disease_model.joblib")
symptom_columns = joblib.load("models/symptom_columns.joblib")


class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.get("/")
def home():
    return {"message": "Disease Prediction API is running"}


@app.post("/predict")
def predict(request: SymptomRequest):

    # Create all symptoms with 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=symptom_columns
    )

    # Mark provided symptoms as 1
    for symptom in request.symptoms:
        if symptom in symptom_columns:
            input_data.loc[0, symptom] = 1

    prediction = model.predict(input_data)[0]

    return {
        "predicted_disease": prediction
    }


