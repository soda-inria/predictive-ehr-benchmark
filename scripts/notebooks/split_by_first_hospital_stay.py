# %%
import pandas as pd
import polars as pl
from medem.preprocessing.utils import PolarsData
from medem.experiences.cohort import EventCohort
from medem.utils import to_pandas, clean_date_cols, to_lazyframe
from medem.experiences.configurations import (
    CONFIG_PROGNOSIS_COHORT,
    cohort_configuration_to_str,
    TEST_HOSPITALS_DICT,
)
from copy import deepcopy
from medem.constants import *

# %%
database_name = "cse210038_20220921_160214312112"
database = PolarsData(database_name=database_name)

cohort_conf = deepcopy(CONFIG_PROGNOSIS_COHORT)
cohort_conf.pop("target_chapter", None)
cohort_name = cohort_configuration_to_str(cohort_conf)
event_cohort = EventCohort(folder=DIR2DATA / cohort_name, lazy=False)

# %%
target = event_cohort.person
study_start = CONFIG_PROGNOSIS_COHORT["study_start"]
study_end = CONFIG_PROGNOSIS_COHORT["study_end"]
hospital_names_ext_test_set = list(TEST_HOSPITALS_DICT.keys())

# %%
from medem.preprocessing.selection import split_train_test_w_hospital_ids

df_split = split_train_test_w_hospital_ids(
    database=database,
    inclusion_sessions=target,
    study_end=study_end,
    study_start=study_start,
    hospital_names_ext_test_set=hospital_names_ext_test_set,
)

# %%
df_split["most_visited_hospital"].value_counts()

# %%
df_split["dataset"].value_counts()

# %%
