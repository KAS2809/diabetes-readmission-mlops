from pathlib import Path

import joblib

from src.data import load_data, create_target, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_readmission_pipeline.pkl"


def main():
    # Load saved pipeline
    pipeline = joblib.load(MODEL_PATH)

    print("Saved pipeline loaded successfully.")

    # Load data
    df = load_data()
    df = create_target(df)

    _, test_df = split_data(df)

    X_test = test_df.drop(columns=["readmit_binary"])

    # Take one raw patient record
    sample = X_test.iloc[[0]]

    prediction = pipeline.predict(sample)[0]
    probability = pipeline.predict_proba(sample)[0, 1]

    print("\nPrediction:")
    print(prediction)

    print("\nReadmission probability:")
    print(f"{probability:.4f}")


if __name__ == "__main__":
    main()