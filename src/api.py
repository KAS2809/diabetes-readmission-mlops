from fastapi import FastAPI
from pathlib import Path
import pandas as pd

from src.schemas import PatientData
import joblib

app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description=(
        "API for predicting the probability of "
        "30-day hospital readmission for diabetic patients."
    ),
    version="1.0.0",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "diabetes_readmission_pipeline.pkl"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "decision_threshold.pkl"
)


model = joblib.load(MODEL_PATH)
decision_threshold = joblib.load(THRESHOLD_PATH)


@app.get("/")
def root():
    return {
        "message": "Diabetes Readmission Prediction API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "decision_threshold": float(decision_threshold),
    }
@app.post("/predict")
def predict(patient: PatientData):

    patient_dict = patient.model_dump()

    # Convert API-friendly names back to the
    # original dataset column names.
    column_mapping = {
        "glyburide_metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
    }

    for api_name, dataset_name in column_mapping.items():
        patient_dict[dataset_name] = patient_dict.pop(api_name)

    patient_df = pd.DataFrame([patient_dict])

    probability = model.predict_proba(patient_df)[0, 1]

    prediction = int(
        probability >= decision_threshold
    )

    risk_classification = (
        "HIGH RISK"
        if prediction == 1
        else "LOW RISK"
    )

    return {
        "readmission_probability": round(
            float(probability), 4
        ),
        "prediction": prediction,
        "risk_classification": risk_classification,
        "decision_threshold": float(
            decision_threshold
        ),
    }