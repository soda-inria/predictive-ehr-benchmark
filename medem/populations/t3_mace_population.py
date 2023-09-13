from pathlib import Path
from loguru import logger
import numpy as np
import polars as pl

from medem.constants import (
    COLNAME_OUTCOME,
    COLNAME_PERSON,
    DIR2DATA,
    DIR2DOCS_COHORT,
)
from medem.experiences.configurations import (
    CONFIG_MACE_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.preprocessing.selection import (
    create_outcome,
    select_population,
    split_train_test_w_inclusion_start,
)
from medem.preprocessing.utils import (
    I2b2Data,
    add_statics,
    create_event_cohort,
)
from medem.preprocessing.selection import split_train_test_w_hospital_ids
from medem.reports.flowchart import get_flowchart_from_inclusion_ids
from medem.utils import to_lazyframe, to_pandas, to_polars

"""
This script creates the MACE population from the i2b2 database.

The cohort are the patients with a hospitalization (0 days and more ie.
incomplete and complete) visits with at least 2 visits. 

For the dates of the study, I used 2018-01-01 and 2021-12-31 as the study period since after 2021,
there is an unexpected drop in the number of MACE codes: https://soda.gitlabpages.inria.fr/medical_embeddings_transfer/exploration.html see 2023-06-15 remark.

NB: The choice of the index can be done with the parameter index_visit.
"""

if __name__ == "__main__":
    config = CONFIG_MACE_COHORT
    database = I2b2Data(path2tables=config["database_name"])
    cohort_name = cohort_configuration_to_str(config)

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
        index_visit=config["index_visit"],
        lazy_save=True,
    )
    # Saving
    dir2cohort = DIR2DATA / cohort_name
    dir2cohort.mkdir(exist_ok=True, parents=True)
    path2inclusion_criteria = dir2cohort / "inclusion_criteria.parquet"
    logger.info(
        f"\n📃 Save inclusion criteria at {str(path2inclusion_criteria)}"
    )
    selected_population.inclusion_population.to_parquet(
        path2inclusion_criteria, index=False
    )
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
    selected_population.inclusion_ids[
        f"No MACE at index visit ({config.get('index_visit', '')})"
    ] = (target[COLNAME_PERSON].unique().tolist())

    target.to_parquet(dir2cohort / "target.parquet", index=False)
    logger.info("\n📃 Create static table\n---------------------")

    person_static = add_statics(
        inclusion_sessions=target,
        database=database,
    )
    logger.info("\n 🏥 Create train test ix for transfer")
    dataset_split = split_train_test_w_inclusion_start(
        inclusion_sessions=person_static,
        test_size=config["test_size"],
    )
    dataset_split.to_parquet(dir2cohort / "dataset_split.parquet", index=False)

    logger.info("\n📃 Create events table\n---------------------")
    n_min_events = config["n_min_events"]
    person, event = create_event_cohort(
        target=person_static,
        database=database,
        event_config=config["event_tables"],
        study_start=config["study_start"],
        n_min_events=n_min_events,
        lazy=config["lazy"],
    )

    selected_population.inclusion_ids[f"< {n_min_events} events logged"] = (
        person.select(COLNAME_PERSON)
        .unique()
        .collect()
        .to_pandas()[COLNAME_PERSON]
        .tolist()
    )

    selected_population.inclusion_ids[
        f"MACE at {config['horizon_in_days']}"
    ] = (
        person.filter(pl.col(COLNAME_OUTCOME) == 1)
        .select(COLNAME_PERSON)
        .unique()
        .collect()
        .to_pandas()[COLNAME_PERSON]
        .tolist()
    )
    #
    logger.info("\n📃 Create flowchart\n---------------------")

    get_flowchart_from_inclusion_ids(
        inclusion_ids=selected_population.inclusion_ids,
        flowchart_name="flowchart_mace",
        target_name=f"MACE at {config['horizon_in_days']}",
        # dir2cohort=DIR2DOCS_COHORT / cohort_name,
    )

    path2person = dir2cohort / "person.parquet"
    path2event = dir2cohort / "event.parquet"
    n_sample = 4000
    path2person_sample = dir2cohort / f"person_{n_sample}_non_cases.parquet"
    path2event_sample = dir2cohort / f"event_{n_sample}_non_cases.parquet"

    if config["lazy"]:
        logger.info(f"Writing sampled tables to {dir2cohort}")
        person_df = person.collect()
        person_df.write_parquet(path2person)
        # write subset for testing pipelines more easily.
        persons_w_split = person_df.join(
            to_polars(dataset_split), on=COLNAME_PERSON, how="inner"
        )
        test_persons = persons_w_split.filter(
            pl.col("dataset") == "external_test"
        )
        train_persons = persons_w_split.filter(
            pl.col("dataset") != "external_test"
        )
        subset_persons = pl.concat(
            [
                test_persons,
                train_persons.filter(pl.col(COLNAME_OUTCOME) == 1),
                train_persons.filter(pl.col(COLNAME_OUTCOME) == 0).sample(
                    n_sample
                ),
            ]
        ).drop(columns=["hospital_split", "dataset"])
        subset_persons.write_parquet(path2person_sample)
        del person_df
        event_sample = event.join(
            to_lazyframe(subset_persons.select(COLNAME_PERSON)),
            on=COLNAME_PERSON,
            how="inner",
        )
        event_sample_df = event_sample.collect()
        event_sample_df.write_parquet(path2event_sample)
        logger.info(f"Writing full tables to {dir2cohort}")
        # write full population
        del selected_population
        event_df = event.collect()
        event_df.write_parquet(path2event)
        ## sink parquet not working (working with simplier computation graph)
        # person.sink_parquet(path2person)
        # event.sink_parquet(path2event)
    else:
        person.to_parquet(path2person, index=False)
        event.to_parquet(path2event, index=False)
