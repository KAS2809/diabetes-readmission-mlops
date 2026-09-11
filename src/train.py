import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline

from src.data import load_data, create_target, split_data
from src.features import (
    FeatureEngineer,
    RareCategoryGrouper,
    TopCategoryGrouper,
)
from src.preprocessing import build_preprocessor


RARE_COLUMNS = [
    "race",
    "gender",
    "admission_type_simplified",
    "admission_source_simplified",
    "discharge_category",
]


def main():
    # 1. Load and prepare data
    df = load_data()

    print("Original dataset shape:")
    print(df.shape)

    df = create_target(df)

    train_df, test_df = split_data(df)

    X_train = train_df.drop(columns=["readmit_binary"])
    y_train = train_df["readmit_binary"]

    X_test = test_df.drop(columns=["readmit_binary"])
    y_test = test_df["readmit_binary"]

    print("\nBefore feature engineering:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 3. Determine numeric and categorical columns
    preprocessor = build_preprocessor()


    # 5. Create baseline model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    # 6. Build full sklearn pipeline
    full_pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer(),
            ),
            (
                "rare_category_grouping",
                RareCategoryGrouper(
                    columns=RARE_COLUMNS,
                    threshold=0.01,
                ),
            ),
            (
                "top_specialty_grouping",
                TopCategoryGrouper(
                    column="medical_specialty",
                    top_n=10,
                ),
            ),
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print("\nFull pipeline created successfully:")
    print(full_pipeline)

    # Train the complete end-to-end pipeline
    full_pipeline.fit(X_train, y_train)

    print("\nFull pipeline trained successfully.")

    # Make predictions directly from raw test data
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]

    print("\nEnd-to-End Pipeline Results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "diabetes_readmission_pipeline.pkl"

    joblib.dump(full_pipeline, model_path)

    print(f"\nSaved trained pipeline to: {model_path}")

if __name__ == "__main__":
    main()
