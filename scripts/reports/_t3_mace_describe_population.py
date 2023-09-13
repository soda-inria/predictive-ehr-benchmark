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
from copy import deepcopy

from medem.utils import *
from medem.constants import *
from medem.experiences.configurations import CONFIG_MACE_COHORT, cohort_configuration_to_str
from medem.experiences.cohort import EventCohort 
from medem.experiences.utils import get_prognosis_prevalence
from medem.preprocessing.utils import I2b2Data
from medem.preprocessing.tasks import get_los, get_mortality, get_rehospitalizations

from event2vec.event_transformer import build_vocabulary

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %%
#database = PolarsData("cse210038_20220921_160214312112")
config = deepcopy(CONFIG_MACE_COHORT)
cohort_name = cohort_configuration_to_str(config)

dir2report_imgs = DIR2DOCS_COHORT / cohort_name
dir2report_imgs.mkdir(exist_ok=True)
dir2cohort = DIR2DATA / cohort_name
event_cohort = EventCohort(folder=dir2cohort)
print(event_cohort.person.shape)
print(event_cohort.event.shape)
print(cohort_name)
event_cohort.event.head()
targets = event_cohort.person
targets[COLNAME_OUTCOME].sum()

# %%
n_sample = 4000
subsample = pd.read_parquet(dir2cohort/f"person_{n_sample}_non_cases.parquet").merge(
    pd.read_parquet(dir2cohort/"dataset_split.parquet"),on=COLNAME_PERSON, how="inner")

train_subsample = subsample.loc[subsample["dataset"]=="train"]

test_subsample = subsample.loc[subsample["dataset"]=="external_test"]
print("train", len(train_subsample), "test", len(test_subsample))

# %% [markdown]
# # Check for leakage

# %%
# check that target_visit_datetime is after followup_start
mask_target_before_followup_start = (targets["target_condition_datetime"] <= targets["followup_start"])
patients_with_target_before_followup_start = targets[mask_target_before_followup_start]
print(len(patients_with_target_before_followup_start)) # linked to visits that are overlapping and billing codes that begin very close to the followup start. 

# check that targets are before 360 days
mask_target_after_horizon = ((targets["target_condition_datetime"] - targets["followup_start"]).dt.total_seconds() / (24*3600)) > 360
patients_with_target_after_horizon = targets[mask_target_after_horizon] # linked to conditions occuring after a visits beginning close to horizon
print(len(patients_with_target_after_horizon))

# check that events in the observation period occurs before followup start
mask_events_after_followup_start = (event_cohort.event["start"] > event_cohort.event["followup_start"])
events_after_followup_start = event_cohort.event[mask_events_after_followup_start]
print(len(events_after_followup_start))

# %% [markdown]
# ## train/test sets

# %%
train_test = pd.read_parquet(dir2cohort/"dataset_split.parquet")
train_test["dataset"].value_counts()

# %%
# Nb patients with MACE in train and test sets
transfer_setup_person = targets.merge(train_test, on=COLNAME_PERSON, how="inner")
mask_train =  transfer_setup_person["dataset"] == "train"
mask_test =  transfer_setup_person["dataset"] == "external_test"
mask_y = transfer_setup_person[COLNAME_OUTCOME] == 1
n_test_mace_person = transfer_setup_person.loc[mask_test & mask_y].shape[0]
n_test_no_mace_person = transfer_setup_person.loc[mask_test & (~mask_y)].shape[0]

n_train_mace_person = transfer_setup_person.loc[mask_train & mask_y].shape[0]
n_train_no_mace_person = transfer_setup_person.loc[mask_train & (~mask_y)].shape[0]

print(f"Test prevalence: {n_test_mace_person}/{n_test_no_mace_person}={100*n_test_mace_person / n_test_no_mace_person:.2f}%")
print(f"Train prevalence: {n_train_mace_person}/{n_train_no_mace_person}={100*n_train_mace_person / n_train_no_mace_person:.2f}%")

print(f"Train and Test prevalence: {n_test_mace_person+n_train_mace_person}/{n_test_no_mace_person+n_train_no_mace_person}={100*(n_test_mace_person +n_train_mace_person)/ (n_test_no_mace_person+n_train_no_mace_person):.2f}%")

# %%
print("The inclusion period for train is ", transfer_setup_person.loc[mask_train, COLNAME_INCLUSION_EVENT_START].min(), "-", transfer_setup_person.loc[mask_train, COLNAME_INCLUSION_EVENT_START].max())
print("The inclusion period for test is ", transfer_setup_person.loc[mask_test, COLNAME_INCLUSION_EVENT_START].min(), "-", transfer_setup_person.loc[mask_test, COLNAME_INCLUSION_EVENT_START].max())

# %% [markdown]
# # Train test differences
#
#

# %% [markdown]
# # Target description

# %% [markdown]
# ### Typical time temporality

# %%
transfer_setup_person.head(2)

# %%
# Might be a bit cheating but this should be stable across time
transfer_setup_person["delta_to_target"] = (transfer_setup_person["target_condition_datetime"] - transfer_setup_person["followup_start"]).dt.total_seconds() / (3600*24)
#mask_zoom = transfer_setup_person["delta_to_tartmget"] <= 30
sns.histplot(data=transfer_setup_person, x="delta_to_target", hue="dataset", common_norm=False, stat="probability", bins=100)

# %%
print("Train")
display(transfer_setup_person.loc[transfer_setup_person["dataset"] == "train","delta_to_target"].describe().to_frame().transpose())
print("Test")
display(transfer_setup_person.loc[transfer_setup_person["dataset"] == "external_test","delta_to_target"].describe().to_frame().transpose())

# %% [markdown]
# The number of targets that happens at 0 day is abnormally big in the train dataset compared to the test set. It is as if we removed all visits in the external test with that modality. 

# %% [markdown]
# # Features description

# %% [markdown]
# ### What is the year-month of inclusion (defined as the start of the index visit) ? 

# %% [markdown]
# #### Stratified by train/test set

# %%
from medem.preprocessing.quality import time_plot
# overall 
_, ax = time_plot(transfer_setup_person, "inclusion_event_start", label="All patients")
_, ax = time_plot(transfer_setup_person.loc[mask_train], "inclusion_event_start", ax=ax, label="Train")
_, ax = time_plot(transfer_setup_person.loc[mask_test], "inclusion_event_start", ax=ax, label="Test")
plt.legend(title="Set")

# %% [markdown]
# ### By followup date (end of the index visit)

# %%
_, ax = time_plot(transfer_setup_person, "followup_start", label="All patients")
_, ax = time_plot(transfer_setup_person.loc[mask_train], "followup_start", ax=ax, label="Train")
_, ax = time_plot(transfer_setup_person.loc[mask_test], "followup_start", ax=ax, label="Test")
plt.legend(title="Set")

# %% [markdown]
# There is a big issue with december, surely related to incomplete hospitalisation.
#
#

# %%
mask_incomplete_hospitalisation = transfer_setup_person["inclusion_event_source_concept_id"] == "hospitalisation incomplète"
_, ax = time_plot(transfer_setup_person, "followup_start", label="All patients")
_, ax = time_plot(transfer_setup_person.loc[mask_incomplete_hospitalisation], "followup_start", ax=ax, label="Incomplete hospitalisation")
_, ax = time_plot(transfer_setup_person.loc[~mask_incomplete_hospitalisation], "followup_start", ax=ax, label="Complete hospitalisation")
plt.legend(title="Hospitalization status of the index visit")

# %% [markdown]
# The incomplete hospitalization followup was wrong and had been forced at most to 24 hours after the start of the visit. 
#

# %% [markdown]
#
# #### Stratified by MACE at 360 status

# %%
from medem.preprocessing.quality import time_plot
# overall 
_, ax = time_plot(transfer_setup_person, "inclusion_event_start", label="All patients")
plt.legend()
_, ax = time_plot(transfer_setup_person.loc[mask_y], "inclusion_event_start", label="With MACE 360 days after inclusion")
plt.legend()

# %% [markdown]
# ## What are the distribution of the dates of MACE Events ? TODO: prevalence by month

# %%
person_w_mace = transfer_setup_person.loc[mask_y]
target_condition_months_train, _ = time_plot(person_w_mace.loc[mask_train], colname_datetime="target_condition_datetime", label="Train", ax=ax)
target_condition_months_test, ax = time_plot(person_w_mace.loc[mask_test], colname_datetime="target_condition_datetime", label="Test", ax=ax)
target_condition_months_train, _ = time_plot(person_w_mace.loc[mask_train], colname_datetime="target_condition_datetime", label="Train", ax=ax)
target_condition_months_test, ax = time_plot(person_w_mace.loc[mask_test], colname_datetime="target_condition_datetime", label="Test", ax=ax)


label_normalized = "number of MACE normalized"
target_condition_months_train[label_normalized] = target_condition_months_train["count"] / target_condition_months_train["count"].mean()
target_condition_months_test[label_normalized] = target_condition_months_test["count"] / target_condition_months_test["count"].mean()
target_condition_months_train["dataset"] = "Train"
target_condition_months_test["dataset"] = "Test"
#TODO: change for prevalence for month
target_condition_months = pd.concat([target_condition_months_train, target_condition_months_test], axis=0)

# %%
import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
sns.lineplot(data=target_condition_months, x="year-month", y=label_normalized, hue="dataset", ax=axes[0])
axes[0].set(xlabel="Month")
axes[0].set(ylabel="Nb of MACE per month \ndivided by nb of MACE by dataset over the period")
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))

sns.lineplot(data=target_condition_months, x="year-month", y="count", hue="dataset", ax=axes[1])
axes[1].set(
    xlabel="Month",
    ylabel="Nb of MACE per month",
    yscale="log"
)

# %% [markdown]
# There is a big drop of the MACE incidence in the selected cohort after january 2020. It is logical since we only recruit patient with followup during at least one year. This is the same explanation for the rise of number of MACE during the first months, we only reach a stable regime during the 2019 year.

# %% [markdown]
# # Description of the features [TODO]

# %%
for n_min_events in [10, 20, 50, 100]:
    vocabulary = build_vocabulary(event_cohort.event, n_min_events=n_min_events)
    print(n_min_events, len(vocabulary))

# %%
