import json

from src.data import load_data, create_target, split_data


df = load_data()
df = create_target(df)

_, _, test_df = split_data(df)

sample = test_df.drop(
    columns=["readmit_binary"]
).iloc[0]

api_fields = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "diag_1",
    "diag_2",
    "diag_3",
    "payer_code",
    "weight",
    "examide",
    "citoglipton",
]

request_data = sample[api_fields].to_dict()

# Convert NaN values to None for valid JSON.
request_data = {
    key: None if value != value else value
    for key, value in request_data.items()
}

# Rename fields containing hyphens to our
# API-friendly Pydantic field names.
rename_mapping = {
    "glyburide-metformin": "glyburide_metformin",
    "glipizide-metformin": "glipizide_metformin",
    "glimepiride-pioglitazone": "glimepiride_pioglitazone",
    "metformin-rosiglitazone": "metformin_rosiglitazone",
    "metformin-pioglitazone": "metformin_pioglitazone",
}

for old_name, new_name in rename_mapping.items():
    request_data[new_name] = request_data.pop(old_name)

print(
    json.dumps(
        request_data,
        indent=2,
        default=str,
    )
)