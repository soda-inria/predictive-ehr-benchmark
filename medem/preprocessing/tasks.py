from datetime import datetime
from typing import List, Union
from loguru import logger
from medem.preprocessing.utils import (
    PolarsData,
    coarsen_cim10_to_chapter,
    get_cim10_codes,
)

import numpy as np
from medem.constants import (
    COLNAME_CODE,
    COLNAME_DEATH_DATE,
    COLNAME_FOLLOWUP_START,
    COLNAME_LOS,
    COLNAME_LOS_CATEGORY,
    COLNAME_OUTCOME,
    COLNAME_OUTCOME_DATETIME,
    COLNAME_OUTCOME_STAY_ID,
    COLNAME_PERSON,
    COLNAME_SESSION_ID,
    COLNAME_STAY_ID,
    COLNAME_INCLUSION_EVENT_START,
    DIR2RESOURCES,
    LABEL_DCD_DISTINCT,
    LABEL_DCD_INCLUDE,
    MACE_CODES,
)
import pandas as pd
import polars as pl
from medem.utils import (
    clean_date_cols,
    force_datetime,
    to_lazyframe,
    to_pandas,
    to_polars,
)

from sklearn.utils import check_random_state
from sklearn.preprocessing import MultiLabelBinarizer


def get_los(
    inclusion_criteria: pd.DataFrame,
    los_categories: List[float] = [0, 7, 14, np.inf],
) -> pd.DataFrame:
    """Generate los target from a dataframe with inclusion and followup starts.

    Args:
        inclusion_criteria (pd.DataFrame): _description_
        los_categories (List[float], optional): Define los categories in days. Defaults to [0, 7, 14, np.inf].
    Returns:
        pd.DataFrame: _description_
    """
    inclusion_criteria[COLNAME_LOS] = (
        pd.to_datetime(inclusion_criteria[COLNAME_FOLLOWUP_START])
        - pd.to_datetime(inclusion_criteria[COLNAME_INCLUSION_EVENT_START])
    ).dt.days
    # necessary to have sound label (the include lowest yeilds negative first lower bound)
    if los_categories is not None:
        bin_labels = [
            f"({los_categories[i]:.1f}, {los_categories[i+1]:.1f}]"
            for i in range(len(los_categories) - 1)
        ]
        bin_labels[0] = f"[{los_categories[0]:.1f}, {los_categories[1]:.1f}]"
        inclusion_criteria[COLNAME_LOS_CATEGORY] = pd.cut(
            inclusion_criteria[COLNAME_LOS],
            bins=los_categories,
            include_lowest=True,
            labels=bin_labels,
        ).astype(str)
    return inclusion_criteria


def get_rehospitalizations(
    database: PolarsData, inclusion_criteria: pd.DataFrame, horizon_in_days: int
) -> pd.DataFrame:
    visit_occurrence = clean_date_cols(database.visit_occurrence)
    visits_of_interest = visit_occurrence.filter(
        pl.col("visit_source_value").is_in(
            ["hospitalisés", "hospitalisation incomplète"]
        )
    )

    inclusion_criteria_lazy = to_lazyframe(
        inclusion_criteria.rename(
            columns={COLNAME_STAY_ID: COLNAME_STAY_ID + "_index"}
        )
    )

    population_visits = to_pandas(
        visits_of_interest.join(
            inclusion_criteria_lazy, on=COLNAME_PERSON, how="inner"
        )
    )

    # We keep only visits beginning at least half a day after the end of the
    # index hospitalization
    blank_gap_in_days = 0.5
    mask_after_index = population_visits[
        "visit_start_datetime"
    ] >= population_visits[COLNAME_FOLLOWUP_START] + pd.to_timedelta(
        blank_gap_in_days, unit="D"
    )
    visit_eligible_to_target = population_visits[mask_after_index]

    first_following_visit = (
        visit_eligible_to_target.sort_values("visit_start_datetime")
        .groupby(COLNAME_PERSON)
        .first()
    ).reset_index()
    # building the outcome
    ## rehospitalization
    mask_rehospitalization_at_horizon = (
        first_following_visit["visit_start_datetime"]
        - pd.to_timedelta(horizon_in_days, unit="D")
    ) <= first_following_visit[COLNAME_FOLLOWUP_START]
    positive_rehospitalization_at_horizon = first_following_visit[
        mask_rehospitalization_at_horizon
    ].rename(
        columns={
            COLNAME_STAY_ID: "rehospitalization_id",
            "visit_start_datetime": "rehospitalization_datetime",
        }
    )[
        [
            COLNAME_PERSON,
            "rehospitalization_id",
            "rehospitalization_datetime",
        ]
    ]
    # joining the outcomes
    inclusion_criteria_w_rehospitalization = inclusion_criteria.merge(
        positive_rehospitalization_at_horizon,
        on=COLNAME_PERSON,
        how="left",
    ).fillna(
        {
            "rehospitalization_datetime": pd.NaT,
            "rehospitalization_id": 0,
        }
    )
    return inclusion_criteria_w_rehospitalization


def get_mortality(
    database: PolarsData, inclusion_criteria: pd.DataFrame, horizon_in_days: int
) -> pd.DataFrame:
    person = to_pandas(clean_date_cols(database.person))

    inclusion_criteria_w_death = person[
        [COLNAME_PERSON, COLNAME_DEATH_DATE]
    ].merge(inclusion_criteria, on=COLNAME_PERSON, how="inner")
    inclusion_criteria_w_death[COLNAME_DEATH_DATE] = pd.to_datetime(
        inclusion_criteria_w_death[COLNAME_DEATH_DATE]
    )

    mask_dcd_at_horizon = (
        inclusion_criteria_w_death["death_datetime"]
        - pd.to_timedelta(horizon_in_days, unit="D")
    ) <= inclusion_criteria_w_death[COLNAME_FOLLOWUP_START]
    positive_dcd = inclusion_criteria_w_death[mask_dcd_at_horizon].rename(
        columns={"death_datetime": "death_datetime@horizon"}
    )[[COLNAME_PERSON, "death_datetime@horizon"]]
    targets_w_death = inclusion_criteria.merge(
        positive_dcd, on=COLNAME_PERSON, how="left"
    ).fillna({"death_datetime@horizon": pd.NaT})
    targets_w_death["death_datetime@horizon"] = pd.to_datetime(
        targets_w_death["death_datetime@horizon"]
    )
    return targets_w_death


def get_prognosis(
    database: PolarsData,
    inclusion_criteria,
    study_end: datetime,
    cim10_nb_digits: int = 1,
    min_prevalence: float = 0.01,
    random_state: int = 0,
) -> pd.DataFrame:
    """Create the prognosis target for each patient.

    The target visit is one of the visits of the patient:
    - containing at least one cim10 code,
    - selected at random after the first visit.

    The (multi)labels are the billing codes in this visits with the number of classes being defined by:
        - The level of the cim10 hierarchy (cim10_nb_digits)
        - The minimal prevalence of a cim10 code (at the chosen level of hierarchy)
        in the population (min_prevalence)

    Args:
        database (_type_): _description_
        inclusion_criteria (_type_): _description_
        study_end (datetime): Used to limit the potential visits used as targets
        cim10_nb_digits (int, optional): define the level of the hierarchy. Defaults to 2.
        min_prevalence (float, optional): define the minimum prevalence for a given code in the selected hiearchy. Defaults to 0.01.
        random_state (int, optional): random state used for the selection of the target visit for each patient. Defaults to 0.
    Raises:
        ValueError: _description_

    Returns:
        pd .DataFrame: _description_
    """
    condition_occurrence = clean_date_cols(database.condition_occurrence)
    visit_occurrence = clean_date_cols(database.visit_occurrence)

    all_diagnoses = condition_occurrence.join(
        to_lazyframe(
            inclusion_criteria[[COLNAME_PERSON, COLNAME_INCLUSION_EVENT_START]]
        ),
        on=COLNAME_PERSON,
        how="inner",
    )
    diagnoses_in_study = to_pandas(
        all_diagnoses.filter(
            (
                pl.col("condition_start_datetime")
                >= pl.col(COLNAME_INCLUSION_EVENT_START)
            )
            & (pl.col("condition_start_datetime") <= study_end)
        ).drop(columns=[COLNAME_INCLUSION_EVENT_START])
    )

    n_visits_w_cim10_by_patient = (
        diagnoses_in_study.groupby(COLNAME_PERSON)
        .agg(
            **{
                "nb_coded_visits": pd.NamedAgg(
                    "visit_occurrence_id", lambda x: len(np.unique(x))
                )
            }
        )
        .reset_index()
    )
    rg = check_random_state(random_state)
    # Select one visit at random for each patient strictly after its first visit
    n_visits_w_cim10_by_patient = n_visits_w_cim10_by_patient.loc[
        n_visits_w_cim10_by_patient["nb_coded_visits"] > 1
    ]
    n_visits_w_cim10_by_patient.loc[
        :, "target_visit_ix"
    ] = n_visits_w_cim10_by_patient["nb_coded_visits"].apply(
        lambda x: rg.randint(2, x + 1)
    )

    visits_w_diagnoses_in_study = (
        diagnoses_in_study[[COLNAME_STAY_ID, COLNAME_PERSON]]
        .drop_duplicates()
        .merge(n_visits_w_cim10_by_patient, on=COLNAME_PERSON, how="inner")
    )
    visits_w_diagnoses_in_study_w_start_date = to_pandas(
        to_lazyframe(visits_w_diagnoses_in_study).join(
            visit_occurrence.select([COLNAME_STAY_ID, "visit_start_datetime"]),
            on=COLNAME_STAY_ID,
            how="inner",
        )
    )
    visits_w_diagnoses_in_study_w_start_date.sort_values(
        [COLNAME_PERSON, "visit_start_datetime"], inplace=True
    )
    visits_w_diagnoses_in_study_w_start_date["visit_ix"] = (
        visits_w_diagnoses_in_study_w_start_date.groupby(
            COLNAME_PERSON
        ).cumcount()
        + 1
    )
    target_visits = visits_w_diagnoses_in_study_w_start_date.loc[
        visits_w_diagnoses_in_study_w_start_date["target_visit_ix"]
        == visits_w_diagnoses_in_study_w_start_date["visit_ix"]
    ].rename(
        columns={
            "visit_start_datetime": COLNAME_FOLLOWUP_START,
            COLNAME_STAY_ID: COLNAME_OUTCOME_STAY_ID,
        }
    )
    # Get the cim10 codes for each target visit
    target_diagnoses = target_visits[
        [COLNAME_OUTCOME_STAY_ID, "nb_coded_visits", "target_visit_ix"]
    ].merge(
        diagnoses_in_study.rename(
            columns={COLNAME_STAY_ID: COLNAME_OUTCOME_STAY_ID}
        ),
        on=COLNAME_OUTCOME_STAY_ID,
        how="inner",
    )
    # make expection if nb_digits = 1, we use the hierarchy of the cim10 codes to get back the chapter
    if cim10_nb_digits == 1:
        target_diagnoses = coarsen_cim10_to_chapter(
            diagnoses_df=target_diagnoses
        )
        target_diagnoses.dropna(
            subset=["condition_source_value_coarse"], inplace=True
        )
    else:
        target_diagnoses["condition_source_value_coarse"] = target_diagnoses[
            "condition_source_value"
        ].str[: cim10_nb_digits + 1]
    # do not count twice the code for a given stay
    target_diagnoses_deduplicated = target_diagnoses[
        [
            COLNAME_OUTCOME_STAY_ID,
            "condition_source_value_coarse",
            "nb_coded_visits",
            "target_visit_ix",
        ]
    ].drop_duplicates()

    target_vocabulary = (
        target_diagnoses_deduplicated[
            "condition_source_value_coarse"
        ].value_counts()
        / target_visits.shape[0]
    )
    target_vocabulary = target_vocabulary.loc[
        target_vocabulary >= min_prevalence
    ].index
    target_diagnoses_wo_low_prevalence = target_diagnoses_deduplicated.loc[
        target_diagnoses_deduplicated["condition_source_value_coarse"].isin(
            target_vocabulary
        )
    ]
    target_diagnose_grouped_by_stay = (
        target_diagnoses_wo_low_prevalence.groupby(
            [COLNAME_OUTCOME_STAY_ID, "nb_coded_visits", "target_visit_ix"]
        )
        .agg(
            **{
                COLNAME_OUTCOME: pd.NamedAgg(
                    "condition_source_value_coarse", lambda x: list(x)
                )
            }
        )
        .reset_index()
    )
    targets = (
        target_visits[
            [COLNAME_OUTCOME_STAY_ID, COLNAME_PERSON, COLNAME_FOLLOWUP_START]
        ]
        .merge(
            target_diagnose_grouped_by_stay,
            on=COLNAME_OUTCOME_STAY_ID,
            how="inner",
        )
        .merge(inclusion_criteria.drop(columns=COLNAME_FOLLOWUP_START, axis=1))
    )
    eps = pd.Timedelta(seconds=1)
    targets[COLNAME_FOLLOWUP_START] = targets[COLNAME_FOLLOWUP_START] - eps
    return targets


def get_mace(
    database: PolarsData,
    inclusion_criteria: pd.DataFrame,
    horizon_in_days: int = 360,
) -> pd.DataFrame:
    """
    Create MACE target at a given horizon from the followup start.
    It only considers the first MACE in the followup.

    Args:
        database (PolarsData): _description_
        inclusion_criteria (pd.DataFrame): _description_
        horizon_in_days (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    conditions = database.condition_occurrence
    # filter-in before the horizon:
    eligible_conditions = conditions.join(
        to_lazyframe(inclusion_criteria).select(
            [COLNAME_PERSON, COLNAME_FOLLOWUP_START]
        ),
        on=COLNAME_PERSON,
        how="inner",
    ).filter(
        (
            (
                pl.col("condition_start_datetime")
                - pl.col(COLNAME_FOLLOWUP_START)
            ).dt.seconds()
            / (24 * 3600)
        )
        <= horizon_in_days
    )

    mace_conditions_l = []
    for mace_category, cim_10_codes in MACE_CODES.items():
        mace_category_conditions = eligible_conditions.filter(
            pl.col("condition_source_value").str.contains(
                "|".join(cim_10_codes)
            )
        )
        mace_conditions_l.append(
            mace_category_conditions.with_columns(
                pl.lit(mace_category).alias("ICD10 category")
            )
        )
    mace_conditions = get_cim10_codes(
        conditions=eligible_conditions,
        cim10_codes=MACE_CODES,
    )

    mace_conditions_w_start_of_followup = to_polars(inclusion_criteria).join(
        to_polars(
            mace_conditions.select([COLNAME_PERSON, "condition_start_datetime"])
        ),
        on=COLNAME_PERSON,
        how="inner",
    )
    # Exclusion criteria to remove patient that are non MACE incident: these are *too easy targets*
    inclusion_w_preceding_mace = mace_conditions_w_start_of_followup.filter(
        pl.col("condition_start_datetime") <= pl.col(COLNAME_FOLLOWUP_START)
    )

    inclusion_wo_preceding_mace = to_polars(inclusion_criteria).join(
        inclusion_w_preceding_mace[[COLNAME_SESSION_ID]],
        on=COLNAME_SESSION_ID,
        how="anti",
    )
    # filter only conditions after the followup
    mace_conditions_after_followup = mace_conditions.filter(
        pl.col("condition_start_datetime") > pl.col(COLNAME_FOLLOWUP_START)
    )

    # Get back the first condition by person with mace among candidates
    candidate_target = (
        (
            mace_conditions_after_followup.with_columns(
                pl.lit(1).alias(COLNAME_OUTCOME)
            )
            .sort("condition_start_datetime")
            .groupby("person_id")
            .agg(
                pl.col(COLNAME_OUTCOME).first(),
                pl.col("condition_start_datetime")
                .first()
                .alias("target_condition_datetime"),
                pl.col(COLNAME_STAY_ID).first().alias(COLNAME_STAY_ID),
                pl.col("ICD10 category").first().alias("ICD10 category"),
            )
        )
        .join(
            database.visit_occurrence.select(
                [COLNAME_STAY_ID, "visit_start_datetime"]
            ),
            on=COLNAME_STAY_ID,
            how="inner",
        )
        .collect()
    )
    # Joining positive outcome
    target_cols = [COLNAME_OUTCOME]
    target_df = (
        to_polars(inclusion_wo_preceding_mace)
        .join(candidate_target, on=COLNAME_PERSON, how="left")
        .to_pandas()
        .fillna({k: 0 for k in target_cols})
    )
    # remove outcome outside of horizon
    target_df[target_cols] = target_df[target_cols].astype(int)
    n_positive_targets = target_df[COLNAME_OUTCOME].sum()
    logger.info(
        f"Prevalence of one rehospit MACE {n_positive_targets:.0f} / {len(target_df):.0f} = {100*n_positive_targets/len(target_df):.2f}%"
    )
    return target_df
