# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: py310_matthieu
#     language: python
#     name: py310_matthieu
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

from medem.utils import *
from medem.constants import *
from medem.experiences.configurations import CONFIG_MACE_COHORT, cohort_configuration_to_str, CONFIG_MACE_ESTIMATION
from medem.experiences.cohort import EventCohort 
from medem.experiences.utils import get_prognosis_prevalence, get_scores
from medem.preprocessing.utils import I2b2Data
from medem.preprocessing.tasks import get_los, get_mortality, get_rehospitalizations
from medem.experiences.features import get_date_details
from medem.utils import add_age

from event2vec.event_transformer import Event2vecPretrained, OneHotEvent
from event2vec.event_transformer import (
    build_vocabulary,
    restrict_to_vocabulary,
    DemographicsTransformer,
)
from medem.experiences.configurations import PATH2SNDS_EMBEDDINGS


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %%
config= {}
config_cohort = CONFIG_MACE_COHORT
config_experience = CONFIG_MACE_ESTIMATION
#logger.info(f"🧐 Running experience for {config_experience}\n-------------")

cohort_name = cohort_configuration_to_str(config_cohort)
# load data
dir2cohort = DIR2DATA / cohort_name
time_hash_xp = str(hash(datetime.now().strftime("%Y%m%d_%H%M%S")))
if config.get("dir2experience", None) is None:
    dir2experience = DIR2EXPERIENCES / (
        "timesplit__" + cohort_name + "_hash_" + time_hash_xp
    )
else:
    dir2experience = Path(config["dir2experience"])
dir2experience.mkdir(exist_ok=True, parents=True)
# TODO: remove if sufficient memory and replace by
# start replacement
# event_cohort = EventCohort(folder=dir2cohort)
n_sample = 4000
person = pd.read_parquet(
    dir2cohort / f"person_{n_sample}_non_cases.parquet"
)
event = pd.read_parquet(dir2cohort / f"event_{n_sample}_non_cases.parquet")
event_cohort = EventCohort(person=person, event=event)
# end replacement
n_person_raw = event_cohort.person.shape[0]
n_event_raw = event_cohort.event.shape[0]
# fix vocabulary depending on full data
logger.info(
    f"Original number of:\n - persons: {n_person_raw}\n - events {n_event_raw}"
)
if config_experience.get("path2vocabulary", None) is not None:
    study_vocabulary = list(
        pd.read_parquet(config_experience["path2vocabulary"]).columns.values
    )
else:
    study_vocabulary = build_vocabulary(
        event=event_cohort.event,
        n_min_events=config_experience["n_min_events"],
    )
restricted_event = restrict_to_vocabulary(
    event=event_cohort.event,
    vocabulary=study_vocabulary,
)
restricted_person = event_cohort.person.merge(
    restricted_event[COLNAME_PERSON].drop_duplicates(),
    on=COLNAME_PERSON,
    how="inner",
)
logger.info(
    f"Restricted number of:\n - persons: {len(restricted_person)}\n - events {len(restricted_event)}"
)
# adding static features
static_features = get_date_details(
    restricted_person, colname_datetime=COLNAME_INCLUSION_EVENT_START
).drop("inclusion_time_of_day", axis=1)
static_features = add_age(
    df=static_features,
    ref_datetime=COLNAME_INCLUSION_EVENT_START,
    colname_age="age_at_inclusion_event_start",
)
restricted_person_w_statics = static_features
# prepare the runs configurations
experience_grid_dict = {
    "estimator_config": config_experience["estimator_config"],
    "featurizer_config": config_experience["featurizer_config"],
    "subtrain_size": config_experience["subtrain_size"],
    "splitting_rs": config_experience["splitting_rs"],
    "colname_demographics": config_experience["colname_demographics"],
    "n_min_events": [config_experience["n_min_events"]],
}
# ( determinist) dataset splits (eg. temporal or by hospital)
dataset_split = pd.read_parquet(dir2cohort / "dataset_split.parquet")
train_person = restricted_person_w_statics.merge(
    dataset_split.loc[dataset_split["dataset"] == "train"],
    on="person_id",
    how="inner",
)
test_person = restricted_person_w_statics.merge(
    dataset_split.loc[dataset_split["dataset"] == "external_test"],
    on="person_id",
    how="inner",
)
# subsample tests to be able to evaluate with RF.
test_person = test_person.sample(10000, random_state=42)
nb_person_after_split = train_person.shape[0] + test_person.shape[0]

# %%
train_person["y"].value_counts()

# %%
