import sys
import numpy as np
import pandas as pd
import pytest
from medem.constants import (
    COLNAME_LOS_CATEGORY,
    COLNAME_OUTCOME,
    COLNAME_OUTCOME_STAY_ID,
)
from medem.preprocessing.tasks import (
    get_mace,
    get_prognosis,
    get_los,
    get_mortality,
    get_rehospitalizations,
)
from medem.utils import make_df, to_lazyframe


from datetime import datetime


def test_get_mace(mock_database, inclusion_criteria):
    target_df = get_mace(
        database=mock_database,
        inclusion_criteria=inclusion_criteria,
        horizon_in_days=360,
    )

    np.testing.assert_equal(target_df[COLNAME_OUTCOME], np.array([1, 0]))
    icd10_categories = target_df["ICD10 category"].values
    np.testing.assert_equal(
        icd10_categories, ["Acute Cerebrovascular Events (Stroke)", None]
    )

    return


def test_get_los(inclusion_criteria):
    # bins should be in time delta format of days, or los converted in days
    los_target = get_los(inclusion_criteria, los_categories=[0.0, 4, np.inf])
    np.testing.assert_equal(
        los_target[COLNAME_LOS_CATEGORY],
        np.array(["[0.0, 4.0]", "(4.0, inf]"]),
    )


@pytest.mark.parametrize(
    "horizon_in_days, expected_mortality_datetime",
    [
        (1, ["NaT", "NaT"]),
        (
            10,
            [
                "NaT",
                datetime.strptime("2021-01-25 01:11:47", "%Y-%m-%d %H:%M:%S"),
            ],
        ),
    ],
)
def test_get_mortality(
    mock_database,
    inclusion_criteria,
    horizon_in_days,
    expected_mortality_datetime,
):
    from medem.preprocessing.tasks import get_mortality

    mortality_target = get_mortality(
        database=mock_database,
        inclusion_criteria=inclusion_criteria,
        horizon_in_days=horizon_in_days,
    )
    np.testing.assert_equal(
        mortality_target["death_datetime@horizon"],
        np.array(expected_mortality_datetime).astype("datetime64[ns]"),
    )


@pytest.mark.parametrize(
    "horizon_in_days, expected_rehospitalization_datetime",
    [
        (1, ["NaT", "NaT"]),
        (10, ["NaT", datetime.strptime("2021-01-21", "%Y-%m-%d")]),
    ],
)
def test_get_rehospitalizations(
    mock_database,
    inclusion_criteria,
    horizon_in_days,
    expected_rehospitalization_datetime,
):
    rehospitalization_target = get_rehospitalizations(
        database=mock_database,
        inclusion_criteria=inclusion_criteria,
        horizon_in_days=horizon_in_days,
    )
    np.testing.assert_equal(
        rehospitalization_target["rehospitalization_datetime"],
        np.array(expected_rehospitalization_datetime).astype("datetime64[ns]"),
    )


@pytest.mark.parametrize(
    "cim10_nb_digits, expected_y",
    [
        (
            1,
            np.array([["2"], ["11"], ["9", "11", "2"]], dtype="object"),
        ),
        (
            2,
            np.array([["C34"], ["K22"], ["I51", "K22", "C34"]], dtype="object"),
        ),
    ],
)
def test_get_prognosis(
    cim10_nb_digits,
    expected_y,
    mock_database,
):
    condition_occurence = make_df(
        """visit_occurrence_id,person_id,condition_start_datetime,condition_source_value
        G,1,2017-01-01,Z5912
        A,1,2021-01-01,C349
        F,1,2021-01-25,B957
        B,2,2021-01-04,F1725
        B,2,2021-01-04,C349
        B,2,2021-01-04,I102
        E,2,2021-01-19,K225
        C,3,2021-01-12,I498
        D,3,2021-01-21,I517
        D,3,2021-01-21,K225
        D,3,2021-01-21,C349
        D,3,2021-01-21,I517"""
    )
    inclusion_criteria = make_df(
        """person_id,visit_occurrence_id,inclusion_event_source_concept_id,inclusion_event_start,followup_start
        1,A,hospitalisés,2017-01-01,2021-01-10
        2,B,hospitalisés,2021-01-04,2021-01-08
        3,C,hospitalisés,2021-01-12,2021-01-18
        """
    )
    mock_database.condition_occurrence = to_lazyframe(condition_occurence)
    prognosis_target = get_prognosis(
        database=mock_database,
        inclusion_criteria=inclusion_criteria,
        study_end=datetime.strptime("2023-01-01", "%Y-%m-%d"),
        cim10_nb_digits=cim10_nb_digits,
        min_prevalence=0.01,
    )
    np.testing.assert_equal(
        prognosis_target[COLNAME_OUTCOME_STAY_ID].values,
        np.array(["A", "E", "D"]),
    )
    np.testing.assert_equal(
        prognosis_target["y"],
        expected_y,
    )
