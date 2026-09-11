import pandas as pd

from src.features import engineer_features, RareCategoryGrouper, TopCategoryGrouper


def test_engineer_features_creates_expected_columns():
    sample = pd.DataFrame(
        {
            "admission_type_id": [1],
            "discharge_disposition_id": [1],
            "admission_source_id": [7],
            "age": ["[50-60)"],
            "number_outpatient": [1],
            "number_inpatient": [2],
            "number_emergency": [0],
            "A1Cresult": ["None"],
            "insulin": ["Steady"],
            "diag_1": ["250.13"],
            "diag_2": ["786"],
            "diag_3": ["530"],
            "metformin": ["No"],
            "repaglinide": ["No"],
            "nateglinide": ["No"],
            "chlorpropamide": ["No"],
            "glimepiride": ["No"],
            "acetohexamide": ["No"],
            "glipizide": ["No"],
            "glyburide": ["No"],
            "tolbutamide": ["No"],
            "pioglitazone": ["No"],
            "rosiglitazone": ["No"],
            "acarbose": ["No"],
            "miglitol": ["No"],
            "troglitazone": ["No"],
            "tolazamide": ["No"],
            "glyburide-metformin": ["No"],
            "glipizide-metformin": ["No"],
            "glimepiride-pioglitazone": ["No"],
            "metformin-rosiglitazone": ["No"],
            "metformin-pioglitazone": ["No"],
        }
    )

    result = engineer_features(sample)

    assert result["age"].iloc[0] == 55
    assert result["total_visits"].iloc[0] == 3
    assert result["prior_admission_flag"].iloc[0] == 1
    assert result["has_A1C"].iloc[0] == 0
    assert result["insulin_flag"].iloc[0] == 1
    assert result["num_diabetes_meds"].iloc[0] == 1

    assert result["diag_1_group"].iloc[0] == "Diabetes"
    assert result["diag_2_group"].iloc[0] == "Respiratory"
    assert result["diag_3_group"].iloc[0] == "Digestive"


def test_rare_category_grouper():
    train_data = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "A",
                "A",
                "B",
            ]
        }
    )

    test_data = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
            ]
        }
    )

    grouper = RareCategoryGrouper(
        columns=["category"],
        threshold=0.30,
    )

    grouper.fit(train_data)

    result = grouper.transform(test_data)

    assert result["category"].iloc[0] == "A"
    assert result["category"].iloc[1] == "Other"


def test_top_category_grouper():
    train_data = pd.DataFrame(
        {
            "medical_specialty": [
                "InternalMedicine",
                "InternalMedicine",
                "Cardiology",
                "Cardiology",
                "Radiology",
                "Neurology",
            ]
        }
    )

    test_data = pd.DataFrame(
        {
            "medical_specialty": [
                "InternalMedicine",
                "Cardiology",
                "Neurology",
            ]
        }
    )

    grouper = TopCategoryGrouper(
        column="medical_specialty",
        top_n=2,
    )

    grouper.fit(train_data)

    result = grouper.transform(test_data)

    assert result["medical_specialty"].iloc[0] == "InternalMedicine"
    assert result["medical_specialty"].iloc[1] == "Cardiology"
    assert result["medical_specialty"].iloc[2] == "Other"