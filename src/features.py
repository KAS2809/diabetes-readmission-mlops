import pandas as pd


ADMISSION_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "NotAvailable",
    6: "Null",
    7: "Trauma",
    8: "NotMapped",
}


AGE_MAPPING = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def map_discharge(value):
    if value in [1, 6]:
        return "Home"

    if value in [2, 3, 4, 5, 8, 22, 23, 24, 27, 30]:
        return "Transfer"

    if value in [13, 14, 19, 20, 21]:
        return "Hospice"

    if value == 11:
        return "Death"

    return "Other"


def map_source(value):
    if value in [1, 2, 3]:
        return "Referral"

    if value == 7:
        return "Emergency"

    if value in [4, 5, 6, 10, 18, 19, 22, 25, 26]:
        return "Transfer"

    if value in [11, 12, 13, 14, 23, 24]:
        return "Birth"

    return "Other"


def map_diagnosis(code):
    try:
        code = float(code)
    except (TypeError, ValueError):
        return "Unknown"

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"

    if 250 <= code < 251:
        return "Diabetes"

    if 460 <= code <= 519 or code == 786:
        return "Respiratory"

    if 520 <= code <= 579 or code == 787:
        return "Digestive"

    return "Other"


def engineer_features(df):
    """
    Apply deterministic feature engineering.

    These transformations do not learn values from the training set.
    They can therefore be applied consistently to train, test,
    and future inference data.
    """

    df = df.copy()

    # Admission type
    df["admission_type_simplified"] = (
        df["admission_type_id"]
        .map(ADMISSION_MAP)
        .fillna("Other")
    )
    df = df.drop(columns=["admission_type_id"])

    # Discharge disposition
    df["discharge_category"] = (
        df["discharge_disposition_id"]
        .apply(map_discharge)
    )
    df = df.drop(columns=["discharge_disposition_id"])

    # Admission source
    df["admission_source_simplified"] = (
        df["admission_source_id"]
        .apply(map_source)
    )
    df = df.drop(columns=["admission_source_id"])

    # Age
    df["age"] = (
        df["age"]
        .astype(str)
        .str.strip()
        .map(AGE_MAPPING)
    )

    # Visit-related features
    df["total_visits"] = (
        df["number_outpatient"]
        + df["number_inpatient"]
        + df["number_emergency"]
    )

    df["prior_admission_flag"] = (
        df["number_inpatient"] > 0
    ).astype(int)

    # A1C availability
    df["has_A1C"] = df["A1Cresult"].apply(
        lambda value: 0
        if pd.isna(value) or value == "None"
        else 1
    )

    # Insulin usage
    df["insulin_flag"] = df["insulin"].apply(
        lambda value: 0
        if pd.isna(value) or value == "No"
        else 1
    )

    # Diagnosis groups
    df["diag_1_group"] = df["diag_1"].apply(map_diagnosis)
    df["diag_2_group"] = df["diag_2"].apply(map_diagnosis)
    df["diag_3_group"] = df["diag_3"].apply(map_diagnosis)

    df = df.drop(
        columns=[
            "diag_1",
            "diag_2",
            "diag_3",
        ]
    )

    return df