import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

ROOT_DIR = Path(
    os.getenv(
        "ROOT_DIR", Path(os.path.dirname(os.path.abspath(__file__))) / ".."
    )
)

DIR2CACHE = Path(".joblib_cache")
DIR2CACHE.mkdir(exist_ok=True)

# PATHS
DIR2DOCS = ROOT_DIR / "docs/source"
DIR2DOCS_STATIC = DIR2DOCS / "_static"
DIR2DOCS_IMG = DIR2DOCS_STATIC / "img"
DIR2DOCS_EXPLORATION = DIR2DOCS_IMG / "exploration"
DIR2DOCS_COHORT = DIR2DOCS_IMG / "cohort"
DIR2DOCS_EXPERIENCES = DIR2DOCS_IMG / "experiences"
DIR2DOCS_COHORT.mkdir(parents=True, exist_ok=True)
DIR2DOCS_EXPLORATION.mkdir(parents=True, exist_ok=True)
DIR2DOCS_EXPERIENCES.mkdir(parents=True, exist_ok=True)
DIR2PAPER_IMG = Path(os.getenv("DIR2PAPER_IMG", ""))

# Default paths
DIR2DATA = ROOT_DIR / "data"
DIR2EMBEDDINGS = DIR2DATA / "embeddings"

# aquisition of the snds embeddings if not already done
path2snds_embeddings = (
    DIR2EMBEDDINGS
    / "snds"
    / "echantillon_mid_grain_r=90-centered2019-12-05_19:11:27.parquet"
)
if not path2snds_embeddings.exists():
    (DIR2EMBEDDINGS / "snds").mkdir(exist_ok=True, parents=True)
    url2snds_embeddings = "https://gitlab.com/strayMat/event2vec/-/raw/main/data/results/snds/echantillon_mid_grain_r=90-centered2019-12-05_19:11:27.parquet?ref_type=heads&inline=false"
    snds_embeddings = pd.read_parquet(url2snds_embeddings)
    snds_embeddings.to_parquet(path2snds_embeddings)

DIR2RESOURCES = DIR2DATA / "resources"
DIR2EXPERIENCES = DIR2DATA / "experiences"
DIR2RESULTS = DIR2DATA / "results"

DIR2TESTS = ROOT_DIR / "tests"

# column name for event
COLNAME_PERSON = "person_id"
COLNAME_STAY_ID = "visit_occurrence_id"
COLNAME_PROVIDER = "provider_id"
COLNAME_START = "start"
COLNAME_END = "end"
COLNAME_CODE = "event_concept_id"
COLNAME_SOURCE_CODE = "event_source_concept_id"
COLNAME_TYPE = "event_type_concept_id"
COLNAME_SOURCE_TYPE = "event_source_type_concept_id"
COLNAME_QUALIFIER = "qualifier_concept_id"
COLNAME_VALUE = "value"
COLNAME_UNIT = "unit_concept_id"
COLNAME_UNIT_SOURCE = "unit_source_value"

# Columns of interests
# For person
COLNAME_GENDER = "gender_concept_id"
COLNAME_BIRTH_DATE = "birth_datetime"
COLNAME_DEATH_DATE = "death_datetime"
COLNAME_LOCATION = "location_id"

## added concepts
COLNAME_INCLUSION_CONCEPT = "inclusion_event_source_concept_id"

COLNAME_INCLUSION_EVENT_START = "inclusion_event_start"  # used for targets
COLNAME_FOLLOWUP_START = "followup_start"
COLNAME_OUTCOME_STAY_ID = "outcome_visit_occurence_stay_id"
COLNAME_OUTCOME_DATETIME = "outcome_datetime"
COLNAME_OUTCOME = "y"
COLNAME_INCLUSION_MONTH = "inclusion_month"
COLNAME_INCLUSION_DAY_OF_WEEK = "inclusion_day_of_week"
COLNAME_INCLUSION_TIME_OF_DAY = "inclusion_time_of_day"
COLNAME_LOS = "length_of_stay"
COLNAME_LOS_CATEGORY = "length_of_stay_category"

COLNAME_DISCHARGE_MONTH = "discharge_month"
COLNAME_DISCHARGE_DAY_OF_WEEK = "discharge_day_of_week"
COLNAME_DISCHARGE_TIME_OF_DAY = "discharge_time_of_day"
OBSERVATION_START = "observation_start"
OBSERVATION_END = "observation_end"
COLNAME_SESSION_ID = "session_id"
COLNAMES_EVENT = [
    COLNAME_PERSON,
    COLNAME_STAY_ID,
    COLNAME_START,
    COLNAME_SOURCE_CODE,
    COLNAME_FOLLOWUP_START,
    COLNAME_SOURCE_TYPE,
]

COLNAME_TARGET_CODE = "target_concept_id"
# TASK names
LABEL_DCD_INCLUDE = "include"
LABEL_DCD_DISTINCT = "distinct"

TASK_MORTALITY = "mortality"
TASK_REHOSPITALIZATION = "rehospitalization"
TASK_LOS = "length_of_stay"
TASK_LOS_CATEGORICAL = "length_of_stay_categorical"
TASK_PROGNOSIS = "prognosis"
TASK_MACE = "MACE"

# ICD10 codes


MACE_CODES = {
    "Acute Myocardial Infarction": [
        "I210",  # "ST elevation (STEMI) myocardial infarction of anterior wall"
        "I211",  # "ST elevation (STEMI) myocardial infarction of inferior wall"
        "I219",  # "ST elevation (STEMI) myocardial infarction, unspecified site",
        "I220",  # "Subsequent ST elevation (STEMI) myocardial infarction of anterior wall",
        "I221",  # "Subsequent ST elevation (STEMI) myocardial infarction of inferior wall",
        "I229",  # "Subsequent ST elevation (STEMI) myocardial infarction, unspecified site",
    ],
    "Unstable Angina": [
        "I200",  # "Unstable angina"
        "I208",  # "Other forms of angina pectoris"
        "I209",  # "Angina pectoris, unspecified"
    ],
    "Acute Heart Failure": [
        "I501",  # "Left ventricular failure"
        "I5020",  # "Unspecified systolic (congestive) heart failure"
        "I5021",  # "Acute systolic (congestive) heart failure"
        "I5022",  # "Chronic systolic (congestive) heart failure"
        "I5023",  # "Acute on chronic systolic (congestive) heart failure"
        "I5030",  # "Unspecified diastolic (congestive) heart failure"
        "I5031",  # "Acute diastolic (congestive) heart failure"
        "I5032",  # "Chronic diastolic (congestive) heart failure"
        "I5033",  # "Acute on chronic diastolic (congestive) heart failure"
    ],
    "Acute Cerebrovascular Events (Stroke)": [
        "I60",  # Subarachnoid hemorrhage
        "I61",  # Intracerebral hemorrhage
        "I62",  # Other non-traumatic intracranial hemorrhages
        "I630",  # "Cerebral infarction due to thrombosis of precerebral arteries",
        "I631",  # "Cerebral infarction due to embolism of precerebral arteries"
        "I632",  # "Cerebral infarction due to unspecified occlusion or stenosis of precerebral arteries",
        "I633",  # "Cerebral infarction due to cerebral venous thrombosis, nonpyogenic",
        "I634",  # "Cerebral infarction due to unspecified occlusion or stenosis of cerebral arteries",
        "I64",  # Stroke, not specified as hemorrhagic or infarct
    ],
    "Other codes from top pathologies": [
        "I24",  # Other acute ischemic heart disease.
        "I23",  # Some recent complications of acute myocardial infarction
    ],
}
