from src.data import load_data, create_target, split_data
from src.features import engineer_features


def main():
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

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    print("\nAfter feature engineering:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("\nSample engineered columns:")
    print(
        X_train[
            [
                "age",
                "total_visits",
                "prior_admission_flag",
                "has_A1C",
                "insulin_flag",
                "diag_1_group",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()