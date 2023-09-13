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
from medem.experiences.configurations import CONFIG_LOS_COHORT, cohort_configuration_to_str
from medem.experiences.cohort import EventCohort 
from medem.experiences.utils import get_prognosis_prevalence
from medem.preprocessing.utils import I2b2Data
from medem.preprocessing.tasks import get_los, get_mortality, get_rehospitalizations
from medem.reports.describe import describe_cohort

from event2vec.event_transformer import build_vocabulary

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %%
#database = PolarsData("cse210038_20220921_160214312112")
config = deepcopy(CONFIG_LOS_COHORT)
cohort_name = cohort_configuration_to_str(config)

dir2report_imgs = DIR2DOCS_COHORT / cohort_name
dir2report_imgs.mkdir(exist_ok=True)
dir2cohort = DIR2DATA / cohort_name
event_cohort = EventCohort(folder=dir2cohort)
print(event_cohort.person.shape)
print(event_cohort.event.shape)
print(cohort_name)
events = event_cohort.event
event_cohort.event.head()
targets = event_cohort.person
targets[COLNAME_OUTCOME].sum()

# %% [markdown]
# # Check for leakage

# %% [markdown]
# ## train/test sets

# %%
train_test = pd.read_parquet(dir2cohort/"dataset_split.parquet")
train_test["dataset"].value_counts()

# %%
# Nb patients with Long Los in train and test sets
transfer_setup_person = targets.merge(train_test, on=COLNAME_PERSON, how="inner")
mask_train =  transfer_setup_person["dataset"] == "train"
mask_test =  transfer_setup_person["dataset"] == "external_test"
mask_y = transfer_setup_person[COLNAME_OUTCOME] == 1
n_test_long_los_person = transfer_setup_person.loc[mask_test & mask_y].shape[0]
n_test_no_long_los_person = transfer_setup_person.loc[mask_test & (~mask_y)].shape[0]

n_train_long_los_person = transfer_setup_person.loc[mask_train & mask_y].shape[0]
n_train_no_long_los_person = transfer_setup_person.loc[mask_train & (~mask_y)].shape[0]

print(f"Test prevalence: {n_test_no_long_los_person}/{mask_test.sum()}={100* n_test_no_long_los_person/ mask_test.sum():.2f}%")
print(f"Train prevalence: {n_train_no_long_los_person}/{mask_train.sum()}={100* n_train_no_long_los_person/ mask_train.sum():.2f}%")

print(f"Train and Test prevalence: {n_test_no_long_los_person+n_train_no_long_los_person}/{len(transfer_setup_person)}={100*(n_test_no_long_los_person+n_train_no_long_los_person)/ (len(transfer_setup_person)):.2f}%")

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
# The number of targets that happens at 0 day is abnormally big in the train dataset compared to the test set. It is as if we removed all visits in the external test with that modality. 

# %% [markdown]
# # Features description

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

# %%
