from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union
import numpy as np

import pandas as pd
import polars as pl
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from medem.constants import (
    COLNAME_DEATH_DATE,
    COLNAME_FOLLOWUP_START,
    COLNAME_INCLUSION_CONCEPT,
    COLNAME_INCLUSION_EVENT_START,
    COLNAME_LOS,
    COLNAME_LOS_CATEGORY,
    COLNAME_OUTCOME,
    COLNAME_OUTCOME_DATETIME,
    COLNAME_OUTCOME_STAY_ID,
    COLNAME_PERSON,
    COLNAME_SESSION_ID,
    COLNAME_STAY_ID,
    DIR2DOCS_COHORT,
    DIR2RESOURCES,
    LABEL_DCD_DISTINCT,
    LABEL_DCD_INCLUDE,
    TASK_LOS,
    TASK_LOS_CATEGORICAL,
    TASK_MACE,
    TASK_MORTALITY,
    TASK_REHOSPITALIZATION,
    TASK_PROGNOSIS,
    DIR2CACHE,
)
from medem.preprocessing.tasks import (
    get_los,
    get_mace,
    get_mortality,
    get_prognosis,
    get_rehospitalizations,
)
from medem.preprocessing.utils import (
    PolarsData,
    get_datetime_from_visit_detail,
    merge_visits_into_session,
    sessionize_visits,
)
from medem.utils import (
    add_age,
    clean_date_cols,
    to_lazyframe,
    to_pandas,
    to_polars,
)


# Memory(DIR2CACHE, verbose=0) / @memory.cache
@dataclass
class SelectedPopulation:
    inclusion_population: pd.DataFrame
    inclusion_ids: Dict[str, List[str]]


def select_population(
    database: PolarsData,
    study_start: datetime = parse("2017-01-01"),
    study_end: datetime = parse("2022-06-01"),
    min_age_at_admission: float = 18,
    sup_quantile_visits: float = 0.95,
    n_min_visits: int = 1,
    visits_w_billing_codes_only: bool = False,
    horizon_in_days: int = 90,
    with_incomplete_hospitalization: bool = False,
    index_visit: str = "random",
    max_time_delta_day: float = 1,
    lazy_save: bool = False,
) -> SelectedPopulation:
    """
    Select the population of patients and define one index visit by patient.
    At each step, it saves the ids of the patients that are kept in the population in the inclusion_ids Dict.

    The selection is done in several steps:
    1 - Filter-in on visit type: keep either "hospitalisés" or "hospitalisation incomplète"
    2 - Filter-in on study period: keep only visits that started during the study period
    3 - Filter-in visits with visit end datetime: keep only visits with a visit end datetime, try to get back some end datetimes from visit details
    4 - Merge close visits into sessions (closer than max_time_delta_day)
    5 - Filter-in adult sessions: keep only sessions with patient aged more than `min_age_at_admission`
    6 - Filter-in sessions in horizon: keep only sessions with an horizon of `horizon_in_days` after the end of the sessions
    7 - Filter-out in-hospital deaths: keep only sessions with no death during the hospitalization
    8 - Filter-in sessions with billing codes: keep only sessions with at least one billing code (ccam or cim10)
    9 - Filter-out sessions with too many sessions: keep only sessions with less than `sup_quantile_visits` sessions
    10 - Filter-out sessions with too few sessions: keep only sessions with more than `n_min_visits` sessions

    Finally, keep the first visit of each patient as the index visit.


    Args:
        database (HiveData): _description_
        study_start (datetime, optional): _description_. Defaults to parse("2017-01-01").
        study_end (datetime, optional): _description_. Defaults to parse("2022-06-01").
        min_age_at_admission (float, optional): _description_. Defaults to 18.
        sup_quantile_visits (float, optional): Threshold on numer of visits to exclude outliers. Defaults to 0.95.
        min_visits (int, optional): Minimum number of visit necessary to be included. Defaults to 1.
        visit_w_billing_codes (bool, optional): If True, only consider visits with billing codes. Defaults to False.
        horizon_in_days (int, optional): Number of days necessary to have after
        the end of the hospitalization, Defaults to 90.
        flowchart (str, optional): _description_. Defaults to None.
        lazy_save (bool, optional): Lazy saving for filtering on visits with billing codes.
    Returns:
        _type_: _description_
    """
    # clean table:
    visit_occurence = clean_date_cols(database.visit_occurrence)
    person = clean_date_cols(database.person).collect().to_pandas()
    visit_details = clean_date_cols(database.visit_detail)
    condition_occurrence = clean_date_cols(database.condition_occurrence)

    logger.info("1 - Filter on visit source type")
    visit_source_values = ["hospitalisés"]
    if with_incomplete_hospitalization:
        visit_source_values.append("hospitalisation incomplète")
    # other two types are "consultation externe" and "urgence" but do not have billing codes
    hospital_visits = visit_occurence.filter(
        pl.col("visit_source_value").is_in(visit_source_values)
    )
    logger.info("2 - Filter on study period")
    hospital_visits_study_lazy = hospital_visits.filter(
        (pl.col("visit_start_datetime") >= study_start)
        & (pl.col("visit_start_datetime") <= study_end)
    )
    # Collect here to avoid lazy evaluation
    hospital_visits_study = hospital_visits_study_lazy.collect()
    if with_incomplete_hospitalization:
        stay_excluded_str = "No Hospitalization + incomplete hospitalization"
    else:
        stay_excluded_str = "No Hospitalization"
    inclusion_ids = {
        "initial": person[COLNAME_PERSON].unique(),
        f"{stay_excluded_str} during {study_start.date()} / {study_end.date()}": hospital_visits_study[
            COLNAME_PERSON
        ].unique(),
    }
    # Different naming of the inclusion is better for code readability but bad for memory consumption
    logger.info("3 - Fix visit end datetime for incomplete hospitalizations")
    ### Get visit end date from visit detail if available
    hospital_visits_study_w_end = get_datetime_from_visit_detail(
        visit_occurrence=to_pandas(hospital_visits_study),
        visit_detail=visit_details,
        colname_visit_end_datetime="visit_end_datetime",
    )
    ### Force hospitalization end up to 24h for incomplete visits
    hospital_visits_study_w_fixed_end = to_polars(
        hospital_visits_study_w_end
    ).with_columns(
        pl.when(pl.col("visit_source_value") == "hospitalisation incomplète")
        .then(pl.col("visit_start_datetime").dt.offset_by("1d"))
        .otherwise(pl.col("visit_end_datetime"))
        .alias("visit_end_datetime")
    )
    inclusion_ids[
        f"Without a visit_end_datetime"
    ] = hospital_visits_study_w_fixed_end[COLNAME_PERSON].unique()

    logger.info("4 - Merge visits into sessions")
    visits_w_session_ids = sessionize_visits(
        hospital_visits_study_w_fixed_end, max_time_delta_day=max_time_delta_day
    )
    included_sessions = to_pandas(
        merge_visits_into_session(visits_w_session_ids)
    )
    del (
        hospital_visits_study,
        hospital_visits_study_w_end,
        hospital_visits_study_w_fixed_end,
        visits_w_session_ids,
    )

    logger.info("5 - Keep adult at inclusion")
    person_w_inclusion = person.merge(
        included_sessions[[COLNAME_PERSON, "visit_start_datetime"]],
        on=COLNAME_PERSON,
        how="inner",
    )
    person_w_age_at_inclusion = add_age(
        person_w_inclusion,
        ref_datetime="visit_start_datetime",
        colname_age="age_at_visit_start_datetime",
    )
    mask_adults_at_inclusion = (
        person_w_age_at_inclusion["age_at_visit_start_datetime"]
        >= min_age_at_admission
    )
    person_age_adults_at_inclusion = person_w_age_at_inclusion.loc[
        mask_adults_at_inclusion
    ]
    included_sessions_adults = included_sessions.merge(
        person_age_adults_at_inclusion[
            [COLNAME_PERSON, COLNAME_DEATH_DATE]
        ].drop_duplicates(),
        on=COLNAME_PERSON,
        how="inner",
    )
    inclusion_ids[
        f"Aged below {min_age_at_admission}"
    ] = included_sessions_adults[COLNAME_PERSON].unique()
    del included_sessions

    logger.info(
        "6 - Filter out sessions without enough horizon to avoid censoring"
    )
    mask_horizon = (
        included_sessions_adults["visit_end_datetime"]
        + pd.to_timedelta(horizon_in_days, unit="D")
        < study_end
    )
    included_stays_w_horizon = included_sessions_adults[mask_horizon]
    inclusion_ids[
        f"Insufficient horizon ({horizon_in_days} days) after discharge"
    ] = included_stays_w_horizon[COLNAME_PERSON].unique()
    del included_sessions_adults

    logger.info("7 - Keep sessions without in-hospital mortality")
    mask_patient_dcd_before_end_of_stay = (
        included_stays_w_horizon[COLNAME_DEATH_DATE]
        <= included_stays_w_horizon["visit_end_datetime"]
    )
    included_sessions_wo_dcd = included_stays_w_horizon[
        ~mask_patient_dcd_before_end_of_stay
    ]
    inclusion_ids[
        f"Deceased before the end of their inclusion stay"
    ] = included_sessions_wo_dcd[COLNAME_PERSON].unique()
    del included_stays_w_horizon

    logger.info("8 - Filter out sessions without billing codes")
    if visits_w_billing_codes_only:
        included_sessions_w_billing_codes = filter_session_on_billing(
            included_stays=included_sessions_wo_dcd,
            condition_occurrence=condition_occurrence,
            lazy_save=lazy_save,
        )
        inclusion_ids[
            f"Stays without associated billing codes"
        ] = included_sessions_w_billing_codes[COLNAME_PERSON].unique()
    else:
        included_sessions_w_billing_codes = included_sessions_wo_dcd
    del included_sessions_wo_dcd

    logger.info("9 - Filter out patients with too many stays")
    n_hospits_per_patients = included_sessions_w_billing_codes.groupby(
        COLNAME_PERSON
    )[COLNAME_SESSION_ID].nunique()

    q_095 = n_hospits_per_patients.quantile(sup_quantile_visits)
    mask_too_many_hospitalization = n_hospits_per_patients > q_095
    included_sessions_wo_outliers = included_sessions_w_billing_codes.merge(
        n_hospits_per_patients[~mask_too_many_hospitalization].index.to_frame(
            index=False
        )[COLNAME_PERSON],
        on=COLNAME_PERSON,
        how="inner",
    )
    inclusion_ids[f">{q_095} stays"] = included_sessions_wo_outliers[
        COLNAME_PERSON
    ].unique()
    del included_sessions_w_billing_codes

    # 10 - Remove patients with less than min_visits
    mask_too_few_hospitalization = n_hospits_per_patients < n_min_visits
    hospitalizations_wo_too_few_visits = included_sessions_wo_outliers.merge(
        n_hospits_per_patients[~mask_too_few_hospitalization].index.to_frame(
            index=False
        )[COLNAME_PERSON],
        on=COLNAME_PERSON,
        how="inner",
    )
    ## Keep only one session per patient: first, last or random
    if index_visit == "first":
        inclusion_criteria = (
            hospitalizations_wo_too_few_visits.sort_values(
                "visit_start_datetime"
            )
            .groupby(COLNAME_PERSON)
            .first()
            .reset_index()
        )
    elif index_visit == "last":
        inclusion_criteria = (
            hospitalizations_wo_too_few_visits.sort_values(
                "visit_start_datetime"
            )
            .groupby(COLNAME_PERSON)
            .last()
            .reset_index()
        )
    elif index_visit == "random":
        inclusion_criteria = (
            hospitalizations_wo_too_few_visits.sort_values(
                "visit_start_datetime"
            )
            .groupby(COLNAME_PERSON)
            .sample(1)
            .reset_index()
        )
    else:
        raise ValueError(
            f"index_visit should be either 'first' or 'last', not {index_visit}"
        )
    inclusion_ids[f"Less than {n_min_visits} stays"] = inclusion_criteria[
        COLNAME_PERSON
    ].unique()
    inclusion_criteria_w_clean_cols = inclusion_criteria[
        [
            COLNAME_PERSON,
            COLNAME_SESSION_ID,
            "visit_source_value",
            "visit_start_datetime",
            "visit_end_datetime",
            "discharge_to_source_value",
            "admitted_from_source_value",
            "is_complete_hospitalized",
            "first_care_site_id",
            "last_care_site_id",
            "number_of_stays",
        ]
    ].rename(
        columns={
            "visit_source_value": COLNAME_INCLUSION_CONCEPT,
            "visit_start_datetime": COLNAME_INCLUSION_EVENT_START,
            "visit_end_datetime": COLNAME_FOLLOWUP_START,
        }
    )
    # for spark compatibility
    inclusion_criteria_w_clean_cols[
        "number_of_stays"
    ] = inclusion_criteria_w_clean_cols["number_of_stays"].astype("Int64")

    return SelectedPopulation(inclusion_criteria_w_clean_cols, inclusion_ids)


# TODO: too many arguments for this wrapper function
# TODO: it would be cleaner to consider a blank period of time before a target visit.
def create_outcome(
    database: PolarsData,
    inclusion_criteria: pd.DataFrame,
    horizon_in_days: int = 90,
    task_name: str = TASK_LOS,
    los_categories: np.array = np.array([0, 7, 14, np.inf]),
    deceased: str = LABEL_DCD_DISTINCT,
    study_end: datetime = parse("2023-12-31"),
    cim10_nb_digits: int = 2,
    min_prevalence: float = 0.01,
    random_state: int = None,
) -> pd.DataFrame:
    """
    The target is defined as :

    - TASK_LOS : the length of stay in days,
    - TASK_LOS_CATEGORICAL : the length of stay in days, binned in `los_categories` categories.
    - TASK_MORTALITY : the mortality status at the horizon.
    - TASK_PROGNOSIS : next visit cim10 prediction.

    Args:
        database (HiveData): _description_
        inclusion_criteria (pd.DataFrame): Dataframe with a column with the
        beginning of the inclusion and a datetime for the beginning of the
        followup, for the beginning of the inclusion.
        horizon_in_days (int, optional): _description_. Defaults
        to 90.
        min_prevalence (float, optional): For prognosis task only. Remove diagnosis
        classes with less than 1% prevalences.
        deceased (str, optional): Either include, or. Defaults to "include".
    Returns:
        pd.DataFrame: _description_
    """
    if task_name == TASK_LOS:
        # add los
        inclusion_criteria = get_los(
            inclusion_criteria=inclusion_criteria, los_categories=los_categories
        )
        targets = inclusion_criteria
        targets[COLNAME_OUTCOME] = targets[COLNAME_LOS]
    elif task_name == TASK_LOS_CATEGORICAL:
        # add los
        inclusion_criteria = get_los(
            inclusion_criteria=inclusion_criteria, los_categories=los_categories
        )
        targets = inclusion_criteria
        le = LabelEncoder()
        targets[COLNAME_OUTCOME] = le.fit_transform(
            targets[COLNAME_LOS_CATEGORY]
        )
    elif task_name == TASK_MORTALITY:
        targets = inclusion_criteria
        targets[COLNAME_OUTCOME] = (
            targets[COLNAME_DEATH_DATE].notnull().astype(int)
        )
        inclusion_criteria = get_mortality(
            database=database,
            inclusion_criteria=inclusion_criteria,
            horizon_in_days=horizon_in_days,
        )
    elif task_name == TASK_PROGNOSIS:
        targets = get_prognosis(
            database=database,
            inclusion_criteria=inclusion_criteria,
            study_end=study_end,
            cim10_nb_digits=cim10_nb_digits,
            min_prevalence=min_prevalence,
            random_state=random_state,
        )
    elif task_name == TASK_MACE:
        targets = get_mace(
            database=database,
            inclusion_criteria=inclusion_criteria,
            horizon_in_days=horizon_in_days,
        )
    elif task_name == TASK_REHOSPITALIZATION:
        # coalesce outcomes: deprecated
        raise NotImplementedError
        inclusion_criteria = get_rehospitalizations(
            database=database,
            inclusion_criteria=inclusion_criteria,
            horizon_in_days=horizon_in_days,
        )
        targets = coalesce_outcomes(inclusion_criteria, deceased=deceased)

    return targets


def filter_session_on_billing(
    included_stays: Union[pd.DataFrame, pl.DataFrame],
    condition_occurrence: pl.LazyFrame,
    lazy_save: bool = False,
) -> pd.DataFrame:
    """
    Only consider sessions having at least one diagnosis code.
    """
    diagnoses_in_study = condition_occurrence.join(
        to_lazyframe(
            included_stays[
                [
                    COLNAME_PERSON,
                    "visit_start_datetime",
                    "visit_end_datetime",
                    COLNAME_SESSION_ID,
                ]
            ]
        ),
        on=COLNAME_PERSON,
        how="inner",
    ).filter(
        (pl.col("condition_start_datetime") >= pl.col("visit_start_datetime"))
        & (pl.col("condition_start_datetime") <= pl.col("visit_end_datetime"))
    )
    # strange oom here (), so checkpointing
    #  for t2 prognosis: I had a sink_parquet error
    if lazy_save:
        diagnoses_in_study.sink_parquet(DIR2CACHE / "diagnoses_in_study")
    else:
        diagnoses_in_study.select(
            COLNAME_SESSION_ID
        ).unique().collect().write_parquet(DIR2CACHE / "diagnoses_in_study")
    diagnoses_in_study_df = pl.read_parquet(DIR2CACHE / "diagnoses_in_study")
    visit_ids_w_diagnostic = (
        diagnoses_in_study_df.select(COLNAME_SESSION_ID).unique().to_pandas()
    )
    visit_w_billing = included_stays.merge(
        visit_ids_w_diagnostic[COLNAME_SESSION_ID],
        on=COLNAME_SESSION_ID,
        how="inner",
    )
    return visit_w_billing


def split_train_test_w_hospital_ids(
    database: PolarsData,
    inclusion_sessions: Union[pd.DataFrame, pl.DataFrame],
    study_start: str,
    study_end: str,
    hospital_names_ext_test_set,
    database_type: str = "omop",
) -> pd.DataFrame:
    """Split the dataset in train and external test sets based on hospital IDs.

    First, we split data into an external test set and a train set.
    This test set includes patients which have visits in a predefined set of hospitals.
    We remove all common patients to the train and external test sets.

    Args:
        df_visit_occurrence (dataframe): OMOP visits dataframe
        df_care_site (dataframe): OMOP care sites dataframe
        trigrams_ext_test_set (List[str]): hospitals short names to keep for the external test set
    Returns:
        A dataframe of person_id and their associated datasets
    """
    visit_df = to_pandas(
        clean_date_cols(database.visit_occurrence)
        .join(
            to_lazyframe(inclusion_sessions[[COLNAME_PERSON]]),
            on=COLNAME_PERSON,
            how="inner",
        )
        .select(
            [
                COLNAME_PERSON,
                COLNAME_STAY_ID,
                "visit_start_datetime",
                "care_site_id",
            ]
        )
    )
    visit_df = visit_df.loc[
        (visit_df["visit_start_datetime"] >= study_start)
        & (visit_df["visit_start_datetime"] <= study_end)
    ]
    if database_type == "omop":
        dir2hierarchy_w_hospital_name = (
            DIR2RESOURCES / "care_site_hierarchy_w_hospital_name.csv"
        )
        caresite_hierarchy = pd.read_csv(dir2hierarchy_w_hospital_name)
    elif database_type == "i2b2":
        dir2hierarchy_w_hospital_name = (
            DIR2RESOURCES / "hospital_codes_i2b2.csv"
        )
        caresite_hierarchy = pd.read_csv(
            dir2hierarchy_w_hospital_name, dtype={"care_site_id": str}
        )
    else:
        raise ValueError(
            f"database_type should be either 'omop' or 'i2b2', got {database_type}"
        )

    visit_df_w_hospital = visit_df[
        [COLNAME_STAY_ID, "care_site_id", COLNAME_PERSON]
    ].merge(
        caresite_hierarchy[["care_site_id", "hospital_name"]],
        on="care_site_id",
        how="inner",
    )

    # 1) Split data in train and external test
    mask_test_set = visit_df_w_hospital["hospital_name"].isin(
        hospital_names_ext_test_set
    )
    df_train = visit_df_w_hospital[~mask_test_set]
    df_ext_test = visit_df_w_hospital[mask_test_set]

    df_common = df_train.merge(
        df_ext_test.drop_duplicates(subset=["person_id"])["person_id"],
        on="person_id",
        how="inner",
    )

    # convert to allow __iter__()
    list_common_patients = list(df_common["person_id"].unique())

    df_train_wo_common_patients = df_train[
        ~df_train["person_id"].isin(list_common_patients)
    ]
    df_ext_test_wo_common_patients = df_ext_test[
        ~df_ext_test["person_id"].isin(list_common_patients)
    ]

    df_train_w_most_common_hospit = (
        df_train_wo_common_patients.groupby("person_id")["hospital_name"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    df_ext_test_w_most_common_hospit = (
        df_ext_test_wo_common_patients.groupby("person_id")["hospital_name"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    df_train_w_most_common_hospit.loc[:, "dataset"] = "train"
    df_ext_test_w_most_common_hospit.loc[:, "dataset"] = "external_test"

    split = pd.concat(
        [
            df_train_w_most_common_hospit[
                ["person_id", "hospital_name", "dataset"]
            ],
            df_ext_test_w_most_common_hospit[
                ["person_id", "hospital_name", "dataset"]
            ],
        ]
    ).rename(columns={"hospital_name": "hospital_split"})
    return split


def split_train_test_w_inclusion_start(
    inclusion_sessions: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Split the dataset in train and test sets based on the inclusion start date.
    The hospital id of the first stay of every session is kept for cross-fitting.
    """
    first_inclusion_date = inclusion_sessions[
        COLNAME_INCLUSION_EVENT_START
    ].min()
    last_inclusion_date = inclusion_sessions[
        COLNAME_INCLUSION_EVENT_START
    ].max()
    inclusion_period_days = (last_inclusion_date - first_inclusion_date).days
    temporal_cut = first_inclusion_date + pd.to_timedelta(
        inclusion_period_days * (1 - test_size), unit="day"
    )
    mask_train = (
        inclusion_sessions[COLNAME_INCLUSION_EVENT_START] <= temporal_cut
    )
    mask_test = inclusion_sessions[COLNAME_INCLUSION_EVENT_START] > temporal_cut

    split = inclusion_sessions[[COLNAME_PERSON, "first_care_site_id"]].copy()
    split.loc[mask_train, "dataset"] = "train"
    split.loc[mask_test, "dataset"] = "external_test"
    # force random hospital id if no id present
    split = split.fillna({"first_care_site_id": 8312006712.0})
    return split.rename(columns={"first_care_site_id": "hospital_split"})
