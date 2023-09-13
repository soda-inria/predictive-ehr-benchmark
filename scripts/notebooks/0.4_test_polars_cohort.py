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
from loguru import logger
import numpy as np

from medem.constants import COLNAME_PERSON, DIR2DATA
from medem.experiences.configurations import (
    CONFIG_PROGNOSIS_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.preprocessing.selection import (
    create_outcome,
    select_population,
)
from medem.preprocessing.utils import (
    PolarsData,
    add_statics,
    create_event_cohort,
)
from medem.preprocessing.selection import split_train_test_w_hospital_ids


# %%
config = CONFIG_PROGNOSIS_COHORT
database_name = config["database_name"]
database = PolarsData(database_name)
cohort_name = cohort_configuration_to_str(config)

logger.info("\n🏥 Population creation\n---------------------")
inclusion_criteria = select_population(
    database=database,
    study_start=config["study_start"],
    study_end=config["study_end"],
    min_age_at_admission=config["min_age_at_admission"],
    horizon_in_days=config["horizon_in_days"],
    sup_quantile_visits=config["sup_quantile_visits"],
    n_min_visits=config["n_min_visits"],
    visits_w_billing_codes_only=config["visits_w_billing_codes_only"],
    with_incomplete_hospitalization=config["with_incomplete_hospitalization"],
    # flowchart_name=cohort_name + ".svg",
)
# Saving
dir2cohort = DIR2DATA / cohort_name
dir2cohort.mkdir(exist_ok=True, parents=True)
path2inclusion_criteria = dir2cohort / "inclusion_criteria.parquet"
logger.info(f"\n📃 Save inclusion criteria at {str(path2inclusion_criteria)}")
inclusion_criteria.to_parquet(path2inclusion_criteria, index=False)
logger.info("\n🎯 Target creation\n---------------------")
target = create_outcome(
    database=database,
    inclusion_criteria=inclusion_criteria,
    horizon_in_days=config.get("horizon_in_days", 30),
    task_name=config.get("task_name", None),
    los_categories=config.get("los_categories", np.array([0, 7, np.inf])),
    deceased=config.get("deceased", False),
    study_end=config["study_end"],
    cim10_nb_digits=config.get("cim10_n_digits", 1),
    min_prevalence=config.get("min_prevalence", 0.01),
    random_state=config.get("random_state", 0),
)
logger.info("\n📃 Create static table\n---------------------")
print(target.shape)
person_static = add_statics(
    inclusion_sessions=target,
    database=database,
    static_features_list=STATIC_FEATURES_DICT,
)
logger.info("\n 🏥 Create train test ix for transfer between hospitals")
hospital_split = split_train_test_w_hospital_ids(
    database=database,
    inclusion_sessions=person_static,
    study_start=config["study_start"],
    study_end=config["study_end"],
    hospital_names_ext_test_set=list(TEST_HOSPITALS_DICT.keys()),
)

logger.info("\n📃 Create events table\n---------------------")
person, event = create_event_cohort(
    target=person_static,
    database=database,
    event_config=config["event_tables"],
    n_min_events=10,
)

logger.info(f"Writing tables to {dir2cohort}")
path2person = dir2cohort / "person.parquet"
path2event = dir2cohort / "event.parquet"
# person.to_parquet(path2person, index=False)
# event.to_parquet(path2event, index=False)
person

# %%
