from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_OUTCOME,
    COLNAME_PERSON,
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    COLNAME_START,
    COLNAME_VALUE,
)
from medem.experiences.cohort import EventCohort
from medem.experiences.configurations import EVENT_CONFIG
from medem.preprocessing.selection import split_train_test_w_inclusion_start
from medem.preprocessing.utils import (
    get_datetime_from_visit_detail,
    create_event_cohort,
)
import numpy as np
from medem.utils import to_pandas


def test_select_population(inclusion_criteria, study_end):
    # TODO: better testing (need better mock data)
    assert (inclusion_criteria[COLNAME_FOLLOWUP_START] >= study_end).sum() == 0


def test_get_datetime_from_visit_detail(mock_database):
    recovered_visit_detail = get_datetime_from_visit_detail(
        visit_occurrence=to_pandas(mock_database.visit_occurrence),
        visit_detail=mock_database.visit_detail,
        colname_visit_end_datetime="visit_end_datetime",
    )
    assert recovered_visit_detail["visit_end_datetime"].isna().sum() == 0


def test_create_event_cohort_w_mapping(mock_database, inclusion_criteria):
    target = to_pandas(mock_database.person).merge(
        inclusion_criteria, on=COLNAME_PERSON, how="inner"
    )
    # creating outcome a la mano
    target[COLNAME_OUTCOME] = [0, 1]
    test_event_config = {
        k: v
        for k, v in EVENT_CONFIG.items()
        if k in ["procedure_occurrence", "measurement"]
    }
    person, event = create_event_cohort(
        target=target,
        database=mock_database,
        event_config=test_event_config,
        n_min_events=1,
    )
    event_cohort = EventCohort(
        person=person,
        event=event,
    )
    np.testing.assert_array_equal(
        event_cohort.event[COLNAME_SOURCE_CODE],
        ["GFEA004", "A04AA01", "A04AA01", "3249", "3249", "3249", "3249"],
    )


def test_create_event_cohort(mock_event_cohort, study_start):
    assert (mock_event_cohort.event[COLNAME_START] < study_start).sum() == 0
    mask_procedure = (
        mock_event_cohort.event[COLNAME_SOURCE_TYPE] == "procedure_occurrence"
    )
    assert mock_event_cohort.event.loc[mask_procedure].shape[0] == 3


def test_split_train_test_w_inclusion_start(inclusion_criteria):
    split = split_train_test_w_inclusion_start(
        inclusion_criteria, test_size=0.2
    )

    np.testing.assert_array_equal(split["dataset"], ["train", "external_test"])
