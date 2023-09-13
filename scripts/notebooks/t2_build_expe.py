# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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

# %%

# %%

# %%
before_followup = event_cohort.event.merge(
    targets[[COLNAME_PERSON, "outcome_visit_occurence_stay_id"]], on=COLNAME_PERSON, how="inner"
)
print(event_cohort.event.shape)
print(before_followup.shape)
mask_occure_during_outcome_visit = (before_followup[COLNAME_STAY_ID] == before_followup["outcome_visit_occurence_stay_id"])
print(before_followup[mask_occure_during_outcome_visit].shape)
before_followup[mask_occure_during_outcome_visit]
(event_cohort.event[COLNAME_START] >= event_cohort.event[COLNAME_FOLLOWUP_START]).sum()

# %%
vocabulary_ = build_vocabulary(
    event=event_cohort.event,
    n_min_events=50,
)
restricted_event = restrict_to_vocabulary(
    event=event_cohort.event,
    vocabulary=vocabulary_,
)
restricted_event.shape

# %%
embedding = event2vec(
    events=restricted_event,
    window_radius_in_days=30,
    d=50
)

# %%
from sklearn.preprocessing import MultiLabelBinarizer
from medem.experiences.pipelines import NaivePrognosisBaseline

outcome_classes = event_cohort.person[COLNAME_OUTCOME].explode().unique()

mlb = MultiLabelBinarizer(classes=outcome_classes)
mlb.fit(event_cohort.person[COLNAME_OUTCOME])
y_true = mlb.transform(targets[COLNAME_OUTCOME])

# %%
estimator = NaivePrognosisBaseline(event=event_cohort.event)
#estimator.fit(targets[COLNAME_PERSON], targets)
y_prob = estimator.predict_proba(targets[COLNAME_PERSON], mlb=mlb)
scores = get_scores(y_true=y_true, y_prob=y_prob, classes=outcome_classes)

# %% jupyter={"outputs_hidden": true}
scores

# %%
# sanity check about events having the same date as the followup start. 
followup_start_events = event_cohort.event.loc[event_cohort.event["start"] <= event_cohort.event["followup_start"]]
print(followup_start_events.shape)
followup_start_events["event_source_type_concept_id"].value_counts()

# %% [markdown]
# # Describe the T2 cohort

# %% [markdown]
# ## What is the delta of time between the target visit and the inclusion event (first stay in the period)

# %%
delta_target_to_first_stay = (targets[COLNAME_FOLLOWUP_START] - targets[COLNAME_INCLUSION_EVENT_START]).dt.days
np.quantile(delta_target_to_first_stay, q=[0.1, 0.25, 0.5, 0.75, 0.9])

# %% jupyter={"outputs_hidden": true}
ax = delta_target_to_first_stay.hist(bins=50)
ax.set(xlim=(-1, 1500))

# %% [markdown]
# ### check prevalences

# %%
from medem.preprocessing.tasks import get_prognosis_prevalence
prevalences = get_prognosis_prevalence(targets[COLNAME_OUTCOME])
prevalences.

# %%
# forces 
mlb = MultiLabelBinarizer()
mlb.fit(targets[COLNAME_OUTCOME])
y = mlb.transform(targets[COLNAME_OUTCOME])

# %%
prevalences = get_prognosis_prevalence(y, classes=mlb.classes_)
prevalences
#prevalences.to_dict(orient="records")[0]

# %%
