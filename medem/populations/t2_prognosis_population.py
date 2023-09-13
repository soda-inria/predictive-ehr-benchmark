from loguru import logger
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from medem.constants import COLNAME_PERSON, COLNAME_STAY_ID, DIR2DATA
from medem.experiences.configurations import (
    CONFIG_PROGNOSIS_COHORT,
    TEST_HOSPITALS_DICT,
    cohort_configuration_to_str,
    STATIC_FEATURES_DICT,
)
from medem.experiences.pipelines import NaivePrognosisBaseline
from medem.preprocessing.selection import (
    create_outcome,
    select_population,
    split_train_test_w_inclusion_start,
)
from medem.preprocessing.utils import (
    PolarsData,
    add_statics,
    create_event_cohort,
)
from medem.preprocessing.selection import split_train_test_w_hospital_ids
from medem.reports.flowchart import get_flowchart_from_inclusion_ids

if __name__ == "__main__":
    config = CONFIG_PROGNOSIS_COHORT
    database_name = config["database_name"]
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
        n_min_visits=config["n_min_visits"],
        visits_w_billing_codes_only=config["visits_w_billing_codes_only"],
        with_incomplete_hospitalization=config[
            "with_incomplete_hospitalization"
        ],
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
        horizon_in_days=config.get("horizon_in_days", 30),
        task_name=config.get("task_name", None),
        los_categories=config.get("los_categories", np.array([0, 7, np.inf])),
        deceased=config.get("deceased", False),
        study_end=config["study_end"],
        cim10_nb_digits=config.get("cim10_n_digits", 1),
        min_prevalence=config.get("min_prevalence", 0.01),
        random_state=config.get("random_state", 0),
    )
    target_chapter = config.get("target_chapter", None)
    if target_chapter is not None:
        target["y"] = target["y"].map(lambda x: 1 if target_chapter in x else 0)
    target.to_parquet(dir2cohort / "target.parquet", index=False)
    logger.info("\n📃 Create static table\n---------------------")

    person_static = add_statics(
        inclusion_sessions=target,
        database=database,
        # static_features_list=STATIC_FEATURES_DICT,
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

    #
    logger.info("\n📃 Create flowchart\n---------------------")
    get_flowchart_from_inclusion_ids(
        inclusion_ids=selected_population.inclusion_ids,
        flowchart_name="flowchart_prognosis",
    )
    # Some events starts strictly before the followup start (beginning of the targeted visit)
    #  but are associated with this visit number (74 events). I remove them with the following code:
    event_before_followup = event.merge(
        person[[COLNAME_PERSON, "outcome_visit_occurence_stay_id"]],
        on=COLNAME_PERSON,
        how="inner",
    )
    mask_occure_during_outcome_visit = (
        event_before_followup[COLNAME_STAY_ID]
        == event_before_followup["outcome_visit_occurence_stay_id"]
    )
    event = event_before_followup[~mask_occure_during_outcome_visit].drop(
        columns="outcome_visit_occurence_stay_id"
    )
    logger.info("\n📃 Create last diagnosis features\n---------------------")
    # add index stay diagnoses:
    last_diagnosis_estimator = NaivePrognosisBaseline(event=event)
    last_diagnoses = last_diagnosis_estimator.predict(person[COLNAME_PERSON])
    mlb = MultiLabelBinarizer()
    last_diagnoses_binarized = mlb.fit_transform(last_diagnoses)
    # TODO: seem a bit inefficient
    person["index_stay_chapters"] = [row_ for row_ in last_diagnoses_binarized]

    logger.info(f"Writing tables to {dir2cohort}")
    path2person = dir2cohort / "person.parquet"
    path2event = dir2cohort / "event.parquet"
    person.to_parquet(path2person, index=False)
    event.to_parquet(path2event, index=False)
