from pathlib import Path

import joblib

from src.data import load_data, create_target, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_readmission_pipeline.pkl"


def test_saved_pipeline_prediction():
    pipeline = joblib.load(MODEL_PATH)

    df = load_data()
    df = create_target(df)

    _, test_df = split_data(df)

    X_test = test_df.drop(columns=["readmit_binary"])

    sample = X_test.iloc[[0]]

    prediction = pipeline.predict(sample)
    probability = pipeline.predict_proba(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

    assert probability.shape == (1, 2)
    assert 0 <= probability[0, 1] <= 1