from pathlib import Path

import joblib

from src.data import load_data, create_target, split_data


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


def main():
    # Load champion model
    pipeline = joblib.load(MODEL_PATH)

    # Load selected classification threshold
    threshold = joblib.load(THRESHOLD_PATH)

    print("Champion pipeline loaded successfully.")
    print(f"Decision threshold: {threshold:.2f}")

    # Load dataset
    df = load_data()
    df = create_target(df)

    train_df, validation_df, test_df = split_data(df)

    X_test = test_df.drop(
        columns=["readmit_binary"]
    )

    # Select one raw patient
    sample = X_test.iloc[[0]]

    # Get probability from model
    probability = pipeline.predict_proba(
        sample
    )[0, 1]

    # Apply our selected threshold
    prediction = int(
        probability >= threshold
    )

    print("\nReadmission probability:")
    print(f"{probability:.4f}")

    print("\nPrediction:")
    print(prediction)

    if prediction == 1:
        print("Risk classification: HIGH RISK")
    else:
        print("Risk classification: LOW RISK")


if __name__ == "__main__":
    main()