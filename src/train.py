import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
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


def evaluate_model(model_name, pipeline, X_test, y_test):
    """
    Evaluate a trained pipeline on the test set.
    """

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
    }

    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall:    {results['Recall']:.4f}")
    print(f"F1 Score:  {results['F1']:.4f}")
    print(f"ROC-AUC:   {results['ROC-AUC']:.4f}")
    print(f"PR-AUC:    {results['PR-AUC']:.4f}")

    return results

def find_best_threshold(y_true, y_prob):
    """
    Find the classification threshold that maximizes
    F1 score on the validation set.
    """

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:
        predictions = (
            y_prob >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():
    # 1. Load dataset
    df = load_data()

    print("Original dataset shape:")
    print(df.shape)

    # 2. Create target
    df = create_target(df)

    # 3. Train/test split
    train_df, validation_df, test_df = split_data(df)

    X_train = train_df.drop(columns=["readmit_binary"])
    y_train = train_df["readmit_binary"]

    X_validation = validation_df.drop(columns=["readmit_binary"])
    y_validation = validation_df["readmit_binary"]

    X_test = test_df.drop(columns=["readmit_binary"])
    y_test = test_df["readmit_binary"]

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nValidation shape:")
    print(X_validation.shape)

    print("\nTest shape:")
    print(X_test.shape)
    # 4. Define models
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gradient_boosting_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    # 5. Logistic Regression pipeline
    logistic_pipeline = Pipeline(
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
                build_preprocessor(),
            ),
            (
                "model",
                logistic_model,
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    print("Logistic Regression trained successfully.")

    logistic_results = evaluate_model(
        "Logistic Regression",
        logistic_pipeline,
        X_validation,
        y_validation,
    )

    # 6. Random Forest pipeline
    random_forest_pipeline = Pipeline(
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
                build_preprocessor(),
            ),
            (
                "model",
                random_forest_model,
            ),
        ]
    )

    print("\nTraining Random Forest...")

    random_forest_pipeline.fit(
        X_train,
        y_train,
    )

    print("Random Forest trained successfully.")

    random_forest_results = evaluate_model(
        "Random Forest",
        random_forest_pipeline,
        X_validation,
        y_validation,
    )

    # 7. Gradient Boosting pipeline
    gradient_boosting_pipeline = Pipeline(
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
                build_preprocessor(),
            ),
            (
                "model",
                gradient_boosting_model,
            ),
        ]
    )

    print("\nTraining Gradient Boosting...")

    gradient_boosting_pipeline.fit(
        X_train,
        y_train,
    )

    print("Gradient Boosting trained successfully.")

    gradient_boosting_results = evaluate_model(
        "Gradient Boosting",
        gradient_boosting_pipeline,
        X_validation,
        y_validation,
    )

    # 8. Gradient Boosting probability analysis
    gradient_probabilities = (
        gradient_boosting_pipeline
        .predict_proba(X_validation)[:, 1]
    )

    print("\nGradient Boosting probability distribution:")
    print(
        f"Minimum: "
        f"{gradient_probabilities.min():.4f}"
    )
    print(
        f"Maximum: "
        f"{gradient_probabilities.max():.4f}"
    )
    print(
        f"Mean:    "
        f"{gradient_probabilities.mean():.4f}"
    )

    import numpy as np

    print(
        f"Median:  "
        f"{np.median(gradient_probabilities):.4f}"
    )

    print("\nPatients with probability >= 0.50:")
    print(
        (gradient_probabilities >= 0.50).sum()
    )

    print("Total validation patients:")
    print(len(gradient_probabilities))

    # 9. Gradient Boosting threshold analysis
    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    print("\nGradient Boosting Threshold Analysis:")

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Predicted +':<12}"
    )

    for threshold in thresholds:
        threshold_predictions = (
            gradient_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            threshold_predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            threshold_predictions,
        )

        f1 = f1_score(
            y_validation,
            threshold_predictions,
        )

        predicted_positive = (
            threshold_predictions.sum()
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{predicted_positive:<12}"
        )

    best_threshold, best_validation_f1 = find_best_threshold(
        y_validation,
        gradient_probabilities,
    )

    print("\nBest Gradient Boosting threshold:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_validation_f1:.4f}")


    # 10. Final evaluation on untouched test set
    test_probabilities = (
        gradient_boosting_pipeline
        .predict_proba(X_test)[:, 1]
    )

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    final_results = {
        "Accuracy": accuracy_score(
            y_test,
            test_predictions,
        ),
        "Precision": precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            test_predictions,
        ),
        "F1": f1_score(
            y_test,
            test_predictions,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            test_probabilities,
        ),
        "PR-AUC": average_precision_score(
            y_test,
            test_probabilities,
        ),
    }

    print("\nFINAL TEST RESULTS")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {final_results['Accuracy']:.4f}")
    print(f"Precision: {final_results['Precision']:.4f}")
    print(f"Recall:    {final_results['Recall']:.4f}")
    print(f"F1 Score:  {final_results['F1']:.4f}")
    print(f"ROC-AUC:   {final_results['ROC-AUC']:.4f}")
    print(f"PR-AUC:    {final_results['PR-AUC']:.4f}")
    # 10. Save current pipeline
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = (
        model_dir
        / "diabetes_readmission_pipeline.pkl"
    )

    joblib.dump(
        logistic_pipeline,
        model_path,
    )

    print(
        f"\nSaved trained pipeline to: {model_path}"
    )


if __name__ == "__main__":
    main()
