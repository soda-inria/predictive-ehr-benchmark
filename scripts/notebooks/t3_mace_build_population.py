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

from medem.utils import *
from medem.constants import *
from medem.experiences.configurations import (
    CONFIG_PROGNOSIS_COHORT,
    cohort_configuration_to_str,
)
from medem.experiences.cohort import EventCohort
from medem.experiences.utils import get_prognosis_prevalence
from medem.preprocessing.utils import PolarsData, scan_from_hdfs, I2b2Data
from medem.preprocessing.tasks import (
    get_los,
    get_mortality,
    get_rehospitalizations,
)
import os
from dataclasses import dataclass
import pyarrow.dataset as ds
from medem.reports.flowchart import get_flowchart_from_inclusion_ids

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Script the population selection

# %%
from medem.experiences.configurations import (
    CONFIG_MACE_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.preprocessing.selection import (
    filter_session_on_billing,
    SelectedPopulation,
    create_outcome,
    select_population,
    split_train_test_w_hospital_ids
)
from medem.preprocessing.utils import (
    get_datetime_from_visit_detail,
    add_statics,
    create_event_cohort,
)

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
force_computation = False
cohort_name = cohort_configuration_to_str(config)
dir2cohort = DIR2DATA / cohort_name

# %%
if force_computation:
    logger.info("\n🏥 Population creation\n---------------------")
    selected_population = select_population(
        database=database,
        study_start=config["study_start"],
        study_end=config["study_end"],
        min_age_at_admission=config["min_age_at_admission"],
        horizon_in_days=config["horizon_in_days"],
        sup_quantile_visits=config["sup_quantile_visits"],
        n_min_visits=config["n_min_visits"],
        visits_w_billing_codes_only=config["visits_w_billing_codes_only"],
        with_incomplete_hospitalization=config[
            "with_incomplete_hospitalization"
        ],
    )
    # Saving
    dir2cohort.mkdir(exist_ok=True, parents=True)
    path2inclusion_criteria = dir2cohort / "inclusion_criteria.parquet"
    logger.info(
        f"\n📃 Save inclusion criteria at {str(path2inclusion_criteria)}"
    )
    selected_population.inclusion_population.to_parquet(
        path2inclusion_criteria, index=False
    )


else:
    inclusion_population = pd.read_parquet(
        dir2cohort / "inclusion_criteria.parquet"
    )
    selected_population = SelectedPopulation(
        inclusion_population=inclusion_population,
        inclusion_ids={
            "Target population": inclusion_population[COLNAME_PERSON].unique()
        },
    )


# %% [markdown]
# # Create outcome [passing]

# %%
if force_computation:
    logger.info("\n🎯 Target creation\n---------------------")
    target = create_outcome(
        database=database,
        inclusion_criteria=selected_population.inclusion_population,
        horizon_in_days=config.get("horizon_in_days", 30),
        task_name=config.get("task_name", None),
        los_categories=config.get("los_categories", np.array([0, 7, np.inf])),
        deceased=config.get("deceased", False),
        study_end=config["study_end"],
        cim10_nb_digits=config.get("cim10_n_digits", 1),
        min_prevalence=config.get("min_prevalence", 0.01),
        random_state=config.get("random_state", 0),
    )

    target.to_parquet(dir2cohort / "target.parquet", index=False)
else:
    target = pd.read_parquet(dir2cohort / "target.parquet")

selected_population.inclusion_ids["No MACE at 1st visit"] = (
    target[COLNAME_PERSON].unique().tolist()
)

# %%
logger.info("\n📃 Create static table\n---------------------")
person_static = add_statics(
    inclusion_sessions=target,
    database=database,
)

# %%
# create train_test split based on temporality
inclusion_criteria = person_static
first_inclusion_date = inclusion_criteria[
    COLNAME_INCLUSION_EVENT_START
].min()
last_inclusion_date = inclusion_criteria[
    COLNAME_INCLUSION_EVENT_START
].max()
inclusion_period = (last_inclusion_date - first_inclusion_date).days


# %%
mask_train = inclusion_criteria[COLNAME_INCLUSION_EVENT_START] <= first_inclusion_date + pd.to_timedelta(inclusion_period*0.8, unit="day")
mask_test = inclusion_criteria[COLNAME_INCLUSION_EVENT_START] > first_inclusion_date + pd.to_timedelta(inclusion_period*0.8, unit="day")
mask_train.sum()

# %%
mask_test.sum()

# %%
print(person_static.shape)
person_static.head()

# %%
logger.info("\n📃 Create events table\n---------------------")
n_min_events = 10
person, event = create_event_cohort(
    target=person_static,
    database=database,
    event_config=config["event_tables"],
    n_min_events=n_min_events,
    #study_start=config["study_start"],
    lazy=config["lazy"],
)

# %%
selected_population.inclusion_ids[f">= {n_min_events} events logged"] = (
    person.select(COLNAME_PERSON)
    .unique()
    .collect()
    .to_pandas()[COLNAME_PERSON]
    .tolist()
)

selected_population.inclusion_ids[f"MACE at {config['horizon_in_days']}"] = (
    person.filter(pl.col(COLNAME_OUTCOME) == 1)
    .select(COLNAME_PERSON)
    .unique()
    .collect()
    .to_pandas()[COLNAME_PERSON]
    .tolist()
)

# %%
path2person = dir2cohort / "person.parquet"
path2event = dir2cohort / "event.parquet"

# %%
from medem.reports.flowchart import get_flowchart_from_inclusion_ids

get_flowchart_from_inclusion_ids(
    inclusion_ids=selected_population.inclusion_ids,
    flowchart_name="flowchart_mace",
    target_name=f"MACE at {config['horizon_in_days']}",
)

# %%
# collecting person is passing
person_df = person.collect()
person_df.write_parquet(path2person)
del person_df

# %%
event_df = event.collect()
event_df.write_parquet(path2event)

# %% jupyter={"outputs_hidden": true}
# sink parquet does not work
# TODO: might be relevant to cut the computational graph before sinking eg. before create_event_cohort, once we write the target.parquet.

# I cut the graph by loading directly the target_person, then apply the add_statics and create_event_cohort but got the same error
## PanicException: called `Option::unwrap()` on a `None` value
path2person_sink = str(path2person) + "_sink"
path2event_sink = str(path2event) + "_sink"
# person.sink_parquet(str(path2person_sink))
event.sink_parquet(str(path2event_sink))

# %%
