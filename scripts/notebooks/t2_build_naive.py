# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: event2vec-py310
#     language: python
#     name: event2vec-py310
# ---

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
import seaborn as sns 
import matplotlib.pyplot as plt 
import re
from dateutil.parser import parse
from copy import deepcopy
from medem.utils import *
from medem.constants import *
from medem.experiences.configurations import CONFIG_PROGNOSIS_COHORT, cohort_configuration_to_str
from medem.experiences.cohort import EventCohort 
from medem.experiences.utils import get_prognosis_prevalence, get_scores
from medem.preprocessing.utils import PolarsData, coarsen_cim10_to_chapter
from medem.preprocessing.tasks import get_los, get_mortality, get_rehospitalizations

from medem.experiences.pipelines import NaivePrognosisBaseline

pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Load cohort

# %%
config = deepcopy(CONFIG_PROGNOSIS_COHORT)
cohort_name = cohort_configuration_to_str(config)
dir2report_imgs = DIR2DOCS_COHORT / cohort_name
dir2report_imgs.mkdir(exist_ok=True)
dir2cohort = DIR2DATA / cohort_name
event_cohort = EventCohort(folder=dir2cohort)
print(event_cohort.person.shape)
print(event_cohort.event.shape)
event_cohort.event.head()
targets = event_cohort.person
targets.head(2)
# (14247, 18)
# (1026782, 7

# %%
last_diagnosis_estimator = NaivePrognosisBaseline(event=event_cohort.event)
last_diagnoses = last_diagnosis_estimator.predict(
    targets[COLNAME_PERSON]
)

# %%
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
last_diagnoses_binarized = mlb.fit_transform(last_diagnoses)
last_diagnoses_df = pd.DataFrame(
    last_diagnoses_binarized,
    columns = [f"c_{c_}" for c_ in mlb.classes_]
)
last_diagnoses_df

# %%
