from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd
from loguru import logger
from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_PERSON,
    COLNAME_SESSION_ID,
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    COLNAME_START,
    COLNAME_STAY_ID,
    COLNAME_TARGET_CODE,
    COLNAMES_EVENT,
    DIR2RESOURCES,
)
from medem.utils import clean_date_cols, to_lazyframe, to_pandas, to_polars
import polars as pl
import pyarrow.dataset as ds
from event2vec.utils import DataFrameType
from event2vec.event_transformer import build_vocabulary


omop_tables = [
    "care_site",
    "concept",
    "concept_relationship",
    "condition_occurrence",
    "cost",
    "drug_exposure_administration",
    "drug_exposure_prescription",
    "fact_relationship",
    "measurement",
    "note_deid",
    "person",
    "procedure_occurrence",
    "visit_detail",
    "visit_detail_old",
    "visit_occurrence",
    "visit_occurrence_old",
    "vocabulary",
]


def scan_from_hdfs(table, cse_path):
    table_path = cse_path + table
    return pl.scan_ds(ds.dataset(table_path))


class I2b2Data:
    def __init__(self, path2tables: Path, table_names: List[str] = None):
        self.available_tables = []
        if table_names is None:
            table_names = [
                "visit_occurrence",
                "person",
                "concept",
                "care_site",
                "visit_detail",
                "condition_occurrence",
                "procedure_occurrence",
                "drug_exposure",
            ]
        self.path2tables = path2tables
        for table_name in table_names:
            path2table_ = self.path2tables / table_name
            if path2table_.is_dir():
                path2table_ = str(path2table_) + "/*"
            table = pl.scan_parquet(path2table_)
            setattr(self, table_name, table)


class PolarsData:
    def __init__(self, database_name: str, hdfs_path: str = None):
        self.database_name = database_name
        if hdfs_path is None:
            self.cse_path = f"hdfs://bbsedsi/apps/hive/warehouse/bigdata/omop_exports_prod/hive/{database_name}.db/"
        else:
            self.cse_path = hdfs_path + "/" + database_name
        self.available_tables = omop_tables

        for table_name in self.available_tables:
            setattr(self, table_name, self.load_table(table_name))

    def load_table(self, table_name: str):
        return scan_from_hdfs(table_name, self.cse_path)


def get_datetime_from_visit_detail(
    visit_occurrence: pd.DataFrame,
    visit_detail: pl.LazyFrame,
    colname_visit_end_datetime: str,
) -> pl.DataFrame:
    """
    Search visit end into the visit_detail table RSS and recover them for a given dataframe.
    Args:
        df (pl.LazyFrame): _description_
        visit_detail (pl.LazyFrame): _description_
        colname_visit_end_datetime (str): _description_

    Returns:
        _type_: _description_
    """
    mask_missing_visit_end = visit_occurrence[colname_visit_end_datetime].isna()
    df_w_missing_end_visit = visit_occurrence[mask_missing_visit_end]
    df_ = visit_occurrence.copy()
    df_details_w_missing_visit_end = to_pandas(
        to_lazyframe(df_).join(
            to_lazyframe(visit_detail)
            .filter(pl.col("visit_detail_type_source_value") == "RSS")
            .select(["visit_occurrence_id", "visit_detail_end_datetime"]),
            on="visit_occurrence_id",
            how="inner",
        )
    )
    recovered_visit_end = (
        df_details_w_missing_visit_end.sort_values(
            ["visit_occurrence_id", "visit_detail_end_datetime"]
        )
        .groupby("visit_occurrence_id")
        .agg(
            **{
                colname_visit_end_datetime: pd.NamedAgg(
                    "visit_detail_end_datetime", "last"
                )
            }
        )
        .reset_index()
        .dropna(subset=[colname_visit_end_datetime], axis=0)
    )

    completed_visits = df_w_missing_end_visit.drop(
        colname_visit_end_datetime, axis=1
    ).merge(recovered_visit_end, on="visit_occurrence_id", how="inner")

    recovered_df = pd.concat(
        [visit_occurrence[~mask_missing_visit_end], completed_visits]
    ).reset_index()
    n_missing_visit_recovered = len(completed_visits)
    n_wo_visit_end = (
        len(df_details_w_missing_visit_end) - n_missing_visit_recovered
    )
    logger.info(
        f"Recovered {n_missing_visit_recovered}, throw away {n_wo_visit_end} visits without a visit end datetime in visit or visit_datetime table"
    )
    return recovered_df


def add_statics(
    inclusion_sessions: pd.DataFrame,
    database,
) -> pd.DataFrame:
    """Add statics features related to the stay used for inclusion.

    Args:
        person (pd.DataFrame): _description_ database (HiveData): _description_
        static_features_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Remove all unused columns from person table
    unused_person_cols = [
        "person_source_value",
        "row_status_source_value",
        "row_status_concept_id",
        "race_source_value",
        "ethnicity_concept_id",
        "provider_id",
        "race_source_concept_id",
        "care_site_id",
        "ethnicity_source_concept_id",
        "ethnicity_source_value",
        "row_status_source_concept_id",
        "cdm_source",
        "race_concept_id",
        "gender_concept_id",
    ]
    columns_to_remove = set(unused_person_cols).intersection(
        database.person.columns
    )
    person_w_useful_cols = to_pandas(database.person.drop(columns_to_remove))
    person_w_static = person_w_useful_cols.merge(
        inclusion_sessions, on=COLNAME_PERSON, how="inner"
    )

    return person_w_static


def create_event_cohort(
    target: Union[pd.DataFrame, pl.DataFrame],
    database: Union[PolarsData, I2b2Data],
    event_config: Dict[str, Dict],
    study_start: datetime = None,
    n_min_events: int = 10,
    lazy: bool = False,
) -> Tuple[
    Union[pd.DataFrame, pl.LazyFrame], Union[pd.DataFrame, pl.LazyFrame]
]:
    """
    From inclusion criteria (study start and start of followup) for each
    patients, subset the events corresponding only to events present in the
    observation period.

    Args:
        target (pd.DataFrame): _description_
        database (HiveData): _description_
        event_config (Dict[str, Dict]): Table to include as features.

    Returns:
        person, events
    """

    target_ = to_lazyframe(target)
    events_list = []
    for event_table_name, event_conf_ in event_config.items():
        # map columns to be cehr-bert consistent
        if "path2mapping" in event_conf_.keys():
            path2mapping = event_conf_.pop("path2mapping")
            mapping = pd.read_csv(path2mapping, dtype=str)
        else:
            path2mapping = None
        columned_renamed = {v: k for k, v in event_conf_.items()}

        event_table = (
            clean_date_cols(database.__getattribute__(event_table_name))
            .select(
                [
                    COLNAME_PERSON,
                    "visit_occurrence_id",
                    event_conf_[COLNAME_START],
                    event_conf_[COLNAME_SOURCE_CODE],
                ]
            )
            .rename(columned_renamed)
        )

        event_table_population = event_table.join(
            target_.select(
                [
                    COLNAME_PERSON,
                    COLNAME_FOLLOWUP_START,
                ]
            ),
            on=COLNAME_PERSON,
            how="inner",
        )
        # We have to remove all events that occur after the beginning of
        # followup:
        mask_event_before_followup = pl.col(COLNAME_START) <= pl.col(
            COLNAME_FOLLOWUP_START
        )
        if study_start is not None:
            mask_event_after_inclusion = pl.col(COLNAME_START) >= study_start
        else:
            mask_event_after_inclusion = pl.lit(True)
        event_table_filtered = event_table_population.filter(
            mask_event_before_followup & mask_event_after_inclusion
        )
        event_table_filtered = event_table_filtered.with_columns(
            pl.lit(event_table_name).alias(COLNAME_SOURCE_TYPE)
        )
        # Do the mapping if needed:
        if path2mapping is not None:
            event_table_filtered = (
                event_table_filtered.join(
                    to_lazyframe(mapping),
                    on=COLNAME_SOURCE_CODE,
                    how="inner",
                )
                .drop(COLNAME_SOURCE_CODE)
                .rename({COLNAME_TARGET_CODE: COLNAME_SOURCE_CODE})
            )

        events_list.append(
            event_table_filtered.select(
                COLNAMES_EVENT,
            )
        )

    # An alternative is to continue with polars and save the event in append mode.
    event = pl.concat(events_list)
    # Keep frequent events only
    logger.info(f"Keep only frequent events occurring more than {n_min_events}")
    vocabulary = build_vocabulary(
        event=event,
        colname_code=COLNAME_SOURCE_CODE,
        n_min_events=n_min_events,
    )
    event_restricted_to_vocabulary = event.join(
        to_lazyframe(
            pl.DataFrame({COLNAME_SOURCE_CODE: [str(v) for v in vocabulary]})
        ),
        on=COLNAME_SOURCE_CODE,
        how="inner",
    )

    # only keep persons with events
    person_restricted_to_vocabulary = (
        event_restricted_to_vocabulary.select(COLNAME_PERSON)
        .unique()
        .join(target_, on=COLNAME_PERSON, how="inner")
    )

    # n_patient_wo_events = len(target) - len(target_)
    # logger.info(f"Dropped {n_patient_wo_events} patients wo any events.")
    if not lazy:
        person_restricted_to_vocabulary = to_pandas(
            person_restricted_to_vocabulary
        )
        event_restricted_to_vocabulary = to_pandas(
            event_restricted_to_vocabulary
        )
    return person_restricted_to_vocabulary, event_restricted_to_vocabulary


def coarsen_cim10_to_chapter(diagnoses_df: pd.DataFrame) -> pd.DataFrame:
    """Convert icd10 codes to the higher level of the icd10 hierarchy : the chapter level, an integer between 1 and 22."""
    diagnoses_df_ = diagnoses_df.copy()
    cim_10_hierarchy = pd.read_csv(
        DIR2RESOURCES / "icd10_tree.csv", dtype={"concept_chapter": str}
    ).rename(
        columns={
            "concept_code": "condition_source_value",
            "concept_chapter": "condition_source_value_coarse",
        }
    )
    diagnoses_df_["condition_source_value"] = diagnoses_df_[
        "condition_source_value"
    ].str[:4]
    diagnoses_df_ = diagnoses_df_.merge(
        cim_10_hierarchy[
            ["condition_source_value", "condition_source_value_coarse"]
        ],
        on="condition_source_value",
        how="left",
    )
    nb_non_matched_codes = (
        diagnoses_df_["condition_source_value_coarse"].isna().sum()
    )
    logger.info(
        f"Number of non-matched codes: {nb_non_matched_codes} ie. {100*nb_non_matched_codes/len(diagnoses_df_):.2f} %"
    )
    return diagnoses_df_


def get_cim10_codes(
    conditions: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    cim10_codes: Dict[str, List[str]],
) -> pl.LazyFrame:
    """
    Collect all conditions given a dictionnary of ICD10 codes classes.

    Args:
        conditions (Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]): _description_
        icd10_codes (Dict[str, List[str]]): _description_

    Returns:
        pl.LazyFrame: _description_
    """
    conditions_ = to_lazyframe(conditions)
    codes_conditions_l = []
    for mace_category, cim_10_codes in cim10_codes.items():
        mace_category_conditions = conditions_.filter(
            pl.col("condition_source_value").str.contains(
                "|".join(cim_10_codes)
            )
        )
        codes_conditions_l.append(
            mace_category_conditions.with_columns(
                pl.lit(mace_category).alias("ICD10 category")
            )
        )

    mace_conditions = pl.concat(codes_conditions_l)
    return mace_conditions


def sessionize_visits(
    visit_occurrence: DataFrameType, max_time_delta_day=1
) -> pl.DataFrame:
    """Add a session ID to visits. A session is composed of a sequence of separated
    stays (at least one) that are separated by a time delta of less than max_time_delta_day.

    Args:
        visit_occurrence (DataFrameType): _description_
        max_time_delta_day (int, optional): _description_. Defaults to 1.
    """
    visits_pl = to_polars(
        visit_occurrence.filter(
            pl.col("visit_source_value").is_in(
                ["hospitalisés", "hospitalisation incomplète"]
            )
        )
    )
    visits_sorted = visits_pl.sort(by=["person_id", "visit_start_datetime"])
    # first computed delays between stays
    visits_sorted_w_delay_previous = visits_sorted.with_columns(
        (
            (
                pl.col("visit_start_datetime")
                - pl.col("visit_end_datetime").shift()
            ).dt.seconds()
            / (3600 * 24)
        )
        .over("person_id")
        .alias("delta_previous_visit")
    )
    # Then fusion if less than one day passed:
    visits_sorted_w_session = (
        visits_sorted_w_delay_previous.with_columns(
            [
                (pl.col("delta_previous_visit") >= max_time_delta_day).alias(
                    "new_session_mark"
                )
            ]
        )
        .with_columns(
            [
                pl.col("new_session_mark")
                .cumsum()
                .over("person_id")
                .cast(str)
                .fill_null("")
                .alias("session")
            ]
        )
        .with_columns(
            [
                (pl.col("person_id").cast(str) + pl.col("session")).alias(
                    COLNAME_SESSION_ID
                )
            ]
        )
    )
    return visits_sorted_w_session


def merge_visits_into_session(
    visit_occurrence_w_session_id: DataFrameType,
) -> pl.DataFrame:
    """Merge visits into sessions.
    The choices are to take :
    - the visit start and the reason of admission of the first visit
    - the visit end and the reason of discharge of the last visit
    - concatenate and unique the visit source value of all visits

    Args:
        visit_occurrence_w_session_id (DataFrameType): _description_

    Raises:
        TypeError: _description_

    Returns:
        pl.DataFrame: _description_
    """
    if COLNAME_SESSION_ID not in visit_occurrence_w_session_id.columns:
        raise TypeError(
            f"Column {COLNAME_SESSION_ID} not found in visit_occurrence. Sessionize visits first."
        )
    sessions = (
        visit_occurrence_w_session_id.sort(
            COLNAME_PERSON, "visit_start_datetime"
        )
        .groupby([COLNAME_PERSON, COLNAME_SESSION_ID])
        .agg(
            pl.col("admitted_from_source_value")
            .first()
            .alias("admitted_from_source_value"),
            pl.col("visit_source_value")
            .unique(maintain_order=True)
            .alias("visit_source_value"),
            pl.col("care_site_id").first().alias("first_care_site_id"),
            pl.col("care_site_id").last().alias("last_care_site_id"),
            pl.col("discharge_to_source_value")
            .last()
            .alias("discharge_to_source_value"),
            pl.col("visit_start_datetime").min().alias("visit_start_datetime"),
            pl.col("visit_end_datetime").max().alias("visit_end_datetime"),
            pl.col(COLNAME_STAY_ID).count().alias("number_of_stays"),
        )
        .with_columns(
            pl.col("visit_source_value")
            .list.join(" ")
            .alias("visit_source_value")
        )
    )
    sessions_w_hosptalization_indicator = sessions.with_columns(
        pl.col("visit_source_value")
        .str.contains("hospitalisés")
        .alias("is_complete_hospitalized")
    )
    return sessions_w_hosptalization_indicator
