from src.data import create_target, load_data, split_data
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["decision_threshold"] == 0.15

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_predict_endpoint():
    df = load_data()
    df = create_target(df)

    _, _, test_df = split_data(df)

    sample = test_df.drop(
        columns=["readmit_binary"]
    ).iloc[0]

    patient_data = sample.to_dict()

    # Remove fields that are not part of the API request
    patient_data.pop("encounter_id", None)
    patient_data.pop("patient_nbr", None)

    # Convert NaN values to None
    patient_data = {
        key: None if value != value else value
        for key, value in patient_data.items()
    }

    # Convert dataset column names to API-friendly names
    rename_mapping = {
        "glyburide-metformin": "glyburide_metformin",
        "glipizide-metformin": "glipizide_metformin",
        "glimepiride-pioglitazone": "glimepiride_pioglitazone",
        "metformin-rosiglitazone": "metformin_rosiglitazone",
        "metformin-pioglitazone": "metformin_pioglitazone",
    }

    for old_name, new_name in rename_mapping.items():
        patient_data[new_name] = patient_data.pop(old_name)

    response = client.post(
        "/predict",
        json=patient_data,
    )

    assert response.status_code == 200

    data = response.json()

    assert "readmission_probability" in data
    assert "prediction" in data
    assert "risk_classification" in data
    assert "decision_threshold" in data

    assert 0 <= data["readmission_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["risk_classification"] in [
        "LOW RISK",
        "HIGH RISK",
    ]
    assert data["decision_threshold"] == 0.15