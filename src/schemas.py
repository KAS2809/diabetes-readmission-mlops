from pydantic import BaseModel
from typing import Optional


class PatientData(BaseModel):
    race: Optional[str] = None
    gender: Optional[str] = None
    age: str

    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int

    time_in_hospital: int
    medical_specialty: Optional[str] = None

    num_lab_procedures: int
    num_procedures: int
    num_medications: int

    number_outpatient: int
    number_emergency: int
    number_inpatient: int

    number_diagnoses: int

    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None

    metformin: Optional[str] = "No"
    repaglinide: Optional[str] = "No"
    nateglinide: Optional[str] = "No"
    chlorpropamide: Optional[str] = "No"
    glimepiride: Optional[str] = "No"
    acetohexamide: Optional[str] = "No"
    glipizide: Optional[str] = "No"
    glyburide: Optional[str] = "No"
    tolbutamide: Optional[str] = "No"
    pioglitazone: Optional[str] = "No"
    rosiglitazone: Optional[str] = "No"
    acarbose: Optional[str] = "No"
    miglitol: Optional[str] = "No"
    troglitazone: Optional[str] = "No"
    tolazamide: Optional[str] = "No"

    insulin: Optional[str] = "No"

    glyburide_metformin: Optional[str] = "No"
    glipizide_metformin: Optional[str] = "No"
    glimepiride_pioglitazone: Optional[str] = "No"
    metformin_rosiglitazone: Optional[str] = "No"
    metformin_pioglitazone: Optional[str] = "No"

    change: Optional[str] = None
    diabetesMed: Optional[str] = None

    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None