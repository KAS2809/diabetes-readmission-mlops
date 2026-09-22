from fastapi import FastAPI


app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description=(
        "API for predicting the probability of "
        "30-day hospital readmission for diabetic patients."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Diabetes Readmission Prediction API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }