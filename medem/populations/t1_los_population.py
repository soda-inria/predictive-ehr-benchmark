from loguru import logger

from medem.constants import COLNAME_PERSON, DIR2DATA
from medem.experiences.configurations import (
    CONFIG_LOS_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.preprocessing.selection import (
    create_outcome,
    select_population,
    split_train_test_w_hospital_ids,
    split_train_test_w_inclusion_start,
)
from medem.preprocessing.utils import (
    PolarsData,
    add_statics,
    create_event_cohort,
)
from medem.reports.flowchart import get_flowchart_from_inclusion_ids

if __name__ == "__main__":
    config = CONFIG_LOS_COHORT
    database_name = config["database_name"]
    #
    database = PolarsData(database_name)
    cohort_name = cohort_configuration_to_str(config)

    logger.info("\n🏥 Population creation\n---------------------")
    selected_population = select_population(
        database=database,
        study_start=config["study_start"],
        study_end=config["study_end"],
        min_age_at_admission=config["min_age_at_admission"],
        horizon_in_days=config["horizon_in_days"],
        sup_quantile_visits=config["sup_quantile_visits"],
        # flowchart_name=cohort_name + ".svg",
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
        horizon_in_days=config["horizon_in_days"],
        task_name=config["task_name"],
        los_categories=config["los_categories"],
        deceased=config["deceased"],
    )
    target.to_parquet(dir2cohort / "target.parquet", index=False)
    logger.info("\n📃 Create static table\n---------------------")

    person_static = add_statics(
        inclusion_sessions=target,
        database=database,
    )
    logger.info("\n 🏥 Create train test ix for transfer between hospitals")
    hospital_split = split_train_test_w_hospital_ids(
        database=database,
        inclusion_sessions=person_static,
        study_start=config["study_start"],
        study_end=config["study_end"],
        hospital_names_ext_test_set=list(TEST_HOSPITALS_DICT.keys()),
    )
    hospital_split.to_parquet(
        dir2cohort / "hospital_split.parquet", index=False
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
        n_min_events=n_min_events,
    )
    selected_population.inclusion_ids[f"< {n_min_events} events logged"] = (
        person[COLNAME_PERSON].unique().tolist()
    )
    logger.info("\n📃 Create flowchart\n---------------------")
    get_flowchart_from_inclusion_ids(
        inclusion_ids=selected_population.inclusion_ids,
        flowchart_name="flowchart_los",
    )
    logger.info(f"Writing tables to {dir2cohort}")
    path2person = dir2cohort / "person.parquet"
    path2event = dir2cohort / "event.parquet"
    person.to_parquet(path2person, index=False)
    event.to_parquet(path2event, index=False)
