from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Get the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the raw dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"


def load_data():
    """
    Load the diabetes readmission dataset and convert
    '?' values into missing values.
    """
    df = pd.read_csv(DATA_PATH)

    df = df.replace("?", np.nan)

    return df


def create_target(df):
    """
    Create the binary target variable.

    1 = patient was readmitted within 30 days
    0 = patient was not readmitted within 30 days
    """
    df = df.copy()

    df["readmit_binary"] = (
        df["readmitted"] == "<30"
    ).astype(int)

    df = df.drop(
        columns=[
            "readmitted",
            "encounter_id",
            "patient_nbr",
        ]
    )

    return df


def split_data(df):
    """
    Split the dataset into training and testing sets
    while maintaining the target class distribution.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["readmit_binary"],
    )

    return train_df, test_df