# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
import os
from dataclasses import dataclass

from medem.utils import *
from medem.constants import *
from medem.experiences.cohort import EventCohort
from medem.experiences.configurations import (
    CONFIG_MACE_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.preprocessing.selection import (
    create_outcome,
    select_population,
    split_train_test_w_hospital_ids,
    filter_session_on_billing,
    SelectedPopulation,
)
from medem.preprocessing.utils import (
    PolarsData,
    scan_from_hdfs,
    I2b2Data,
    get_datetime_from_visit_detail,
    add_statics,
    create_event_cohort,
    build_vocabulary,
    get_cim10_codes,
)
from medem.preprocessing.quality import time_plot

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Script the population selection

# %%
config = CONFIG_MACE_COHORT
config["horizon_in_days"] = 360

# %%
# USER = os.environ["USER"]
# hdfs_url = f"hdfs://bbsedsi/user/{USER}/mace_cohort/"
# Loading the data
database = I2b2Data(path2tables=config["database_name"])
print(config["database_name"])

# %% [markdown]
# # selection [passing]

# %%
cohort_name = cohort_configuration_to_str(config)
dir2cohort = DIR2DATA / cohort_name
dir2cohort

# %%
# inclusion before looking at MACE
inclusion_population = pd.read_parquet(
    dir2cohort / "inclusion_criteria.parquet"
)
inclusion_population

# %%
# Date random d'inclusion
random_inclusion_dates, random_inclusion_ax = time_plot(
    inclusion_population, colname_datetime="followup_start"
)

# %% [markdown]
# # Create outcome [passing]

# %%
conditions = database.condition_occurrence
eligible_conditions = conditions.join(
    to_lazyframe(inclusion_population), on=COLNAME_PERSON, how="inner"
)

# %%
mace_conditions = get_cim10_codes(eligible_conditions, MACE_CODES)
mace_conditions_df = mace_conditions.filter(
    (pl.col("condition_start_datetime") >= config["study_start"])
    & (pl.col("condition_start_datetime") <= config["study_end"])
).collect()

# %%
mace_in_study_dates, mace_in_study_ax = time_plot(
    mace_conditions_df.to_pandas(), colname_datetime="condition_start_datetime"
)

# %% [markdown]
# ### All MACE conditions (not restricted to our inclusion population)

# %%
all_mace_conditions = (
    get_cim10_codes(conditions, MACE_CODES)
    .filter(
        (pl.col("condition_start_datetime") >= config["study_start"])
        & (pl.col("condition_start_datetime") <= config["study_end"])
    )
    .collect()
)

# %%
all_mace_in_study_dates, all_mace_in_study_ax = time_plot(
    all_mace_conditions.to_pandas(), colname_datetime="condition_start_datetime"
)

# %%
