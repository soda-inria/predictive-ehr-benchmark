"""Pytest configuration module"""
from copy import deepcopy
from dataclasses import dataclass
import os

import pandas as pd
from medem.utils import to_lazyframe, to_pandas
import polars as pl
import pytest
from dateutil.parser import parse

from medem.constants import COLNAME_OUTCOME, COLNAME_PERSON, DIR2DATA
from medem.experiences.cohort import EventCohort
from medem.experiences.configurations import EVENT_CONFIG
from medem.preprocessing.utils import create_event_cohort, PolarsData


from medem.preprocessing.selection import select_population

dir2mock_data = DIR2DATA / "mock_data"


@pytest.fixture(scope="session")
def study_start():
    return parse("2017-01-01")


@pytest.fixture(scope="session")
def study_end():
    return parse("2022-06-01")


min_age_at_admission = 18


@dataclass()
class VisitDataset:
    visit_occurrence: pd.DataFrame


def load_visit():
    """
    Create a minimalistic dataset for the `visit_merging` function.

    Returns
    -------
    visit_dataset : VisitDataset, a dataclass comprised of
    visit_occurence.
    """
    visit_occurrence = pd.DataFrame(
        {
            "visit_occurrence_id": [
                "A",
                "B",
                "BZ",
                "BZZ",
                "C",
                "CZ",
                "D",
                "E",
                "F",
                "G",
            ],
            "person_id": [1, 2, 2, 2, 3, 3, 3, 2, 1, 1],
            "visit_start_datetime": [
                "2021-01-01",
                "2021-01-04",
                "2021-02-10",
                "2024-02-01",
                "2021-01-12",
                "2021-01-14",
                "2021-01-21",
                "2021-01-19",
                "2021-01-25",
                "2017-01-01",
            ],
            "visit_end_datetime": [
                "2021-01-05",
                "2021-01-08",
                "2021-02-11",
                "2024-02-06",
                "2021-01-18",
                "2021-01-14",
                "2021-01-28",
                "2021-01-21",
                "2021-01-27",
                None,
            ],
            "visit_source_value": [
                "hospitalisés",
                "hospitalisés",
                "hospitalisés",
                "hospitalisés",
                "hospitalisés",
                "hospitalisés",
                "hospitalisés",
                "urgence",
                "hospitalisés",
                "hospitalisés",
            ],
            "row_status_source_value": [
                "supprimé",
                "courant",
                "courant",
                "courant",
                "courant",
                "courant",
                "courant",
                "courant",
                "courant",
                "courant",
            ],
            "care_site_id": ["1", "1", "1", "1", "1", "1", "1", "2", "1", "1"],
            "stay_source_value": [
                "SSR",
                "MCO",
                "MCO",
                "MCO",
                "MCO",
                "MCO",
                "MCO",
                "MCO",
                "SSR",
                "SSR",
            ],
            "admitted_from_source_value": [
                "URG",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "URG",
                "URG",
            ],
            "discharge_to_source_value": [
                "URG",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "DOM",
                "URG",
                "URG",
            ],
        }
    )

    for col in ["visit_start_datetime", "visit_end_datetime"]:
        visit_occurrence[col] = pd.to_datetime(visit_occurrence[col])

    return VisitDataset(visit_occurrence=visit_occurrence)


@pytest.fixture(scope="session")
def mock_person():
    return pd.read_csv(dir2mock_data / "person.csv")


@pytest.fixture(scope="session")
def mock_visit_detail():
    visit_detail = pd.DataFrame(
        {
            "visit_occurrence_id": ["A", "B", "B", "C", "D", "E", "F", "G"],
            "person_id": ["999"] * 8,
            "visit_detail_start_datetime": [
                "2021-01-01",
                "2021-01-04",
                "2021-01-04",
                "2021-01-12",
                "2021-01-13",
                "2021-01-19",
                "2021-01-25",
                "2017-01-01",
            ],
            "visit_detail_end_datetime": [
                "2021-01-05",
                "2021-01-08",
                "2021-01-08",
                "2021-01-18",
                "2021-01-14",
                "2021-01-21",
                "2021-01-27",
                "2017-01-03",
            ],
            "visit_detail_type_source_value": [
                "RSS",
                "MOUV",
                "RSS",
                "RSS",
                "RSS",
                "RSS",
                "RSS",
                "RSS",
            ],
        }
    )
    for col in ["visit_detail_start_datetime", "visit_detail_end_datetime"]:
        visit_detail[col] = pd.to_datetime(visit_detail[col])

    return visit_detail


@pytest.fixture(scope="session")
def mock_procedure_occurrence():
    """
    Create a minimalistic dataset for the ccam.

    Returns
    -------
    ccam_dataset: pd.DataFrame
    """
    person_ids = [1, 1, 2, 2, 3, 3]
    procedure_source_values = [
        "A04AA01",
        "DZEA003",
        "GFEA004",
        "EQQF006",
        "A04AA01",
        "A04AA01",
    ]
    procedure_datetimes = pd.to_datetime(
        [
            "2021-01-01",
            "2021-01-14",
            "2021-01-05",
            "2021-01-10",
            "2021-01-15",
            "2021-01-15",
        ]
    )
    visit_occurrence_ids = ["A", "A", "B", "B", "C", "D"]
    procedure_occurrence = pd.DataFrame(
        {
            "person_id": person_ids,
            "procedure_source_value": procedure_source_values,
            "procedure_datetime": procedure_datetimes,
            "visit_occurrence_id": visit_occurrence_ids,
        }
    )
    return procedure_occurrence


@pytest.fixture(scope="session")
def mock_condition_occurrence():
    """
    Create a minimalistic dataset for the ccam.

    Returns
    -------
    condition_occurrence_dataset: pd.DataFrame
    """
    person_ids = [2, 2, 3]
    condition_source_values = [
        "I60",
        "I210",
        "E42",
    ]
    condition_datetimes = pd.to_datetime(
        [
            "2021-02-10",
            "2024-02-01",
            "2021-01-14",
        ]
    )
    visit_occurrence_ids = ["BZ", "BZZ", "CZ"]
    condition_occurrence = pd.DataFrame(
        {
            "person_id": person_ids,
            "condition_source_value": condition_source_values,
            "condition_start_datetime": condition_datetimes,
            "visit_occurrence_id": visit_occurrence_ids,
        }
    )
    return condition_occurrence


@pytest.fixture(scope="session")
def mock_measurement():
    measurement = pd.read_csv(dir2mock_data / "measurement.csv")
    measurement["measurement_datetime"] = pd.to_datetime(
        measurement["measurement_datetime"]
    )
    return measurement


class MockDataset(PolarsData):
    def __init__(
        self,
        person: pl.DataFrame,
        visit_occurrence: pl.DataFrame,
        visit_detail: pl.DataFrame,
        measurement: pl.DataFrame,
        procedure_occurrence: pl.DataFrame,
        condition_occurrence: pl.DataFrame,
    ):
        self.available_tables = {
            "person": person,
            "visit_occurrence": visit_occurrence,
            "visit_detail": visit_detail,
            "measurement": measurement,
            "procedure_occurrence": procedure_occurrence,
            "condition_occurrence": condition_occurrence,
        }
        self.person = person
        self.visit_occurrence = visit_occurrence
        self.visit_detail = visit_detail
        self.measurement = measurement
        self.procedure_occurrence = procedure_occurrence
        self.condition_occurrence = condition_occurrence


@pytest.fixture(scope="session")
def mock_database(
    mock_person,
    mock_visit_detail,
    mock_procedure_occurrence,
    mock_measurement,
    mock_condition_occurrence,
):
    visit_occurrence = load_visit().visit_occurrence
    db = MockDataset(
        person=to_lazyframe(mock_person),
        visit_occurrence=to_lazyframe(visit_occurrence),
        visit_detail=to_lazyframe(mock_visit_detail),
        measurement=to_lazyframe(mock_measurement),
        procedure_occurrence=to_lazyframe(mock_procedure_occurrence),
        condition_occurrence=to_lazyframe(mock_condition_occurrence),
    )
    return db


@pytest.fixture(scope="session")
def inclusion_criteria(mock_database, study_end, study_start):
    selected_population = select_population(
        database=mock_database,
        study_start=study_start,
        study_end=study_end,
        min_age_at_admission=min_age_at_admission,
        index_visit="first"
        # flowchart_name="test.png",
    )
    return selected_population.inclusion_population


@pytest.fixture(scope="session")
def mock_event_cohort(mock_database, inclusion_criteria):
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
    test_event_config_wo_mapping = deepcopy(test_event_config)
    test_event_config_wo_mapping["measurement"].pop("path2mapping", None)

    person, event = create_event_cohort(
        target=target,
        database=mock_database,
        event_config=test_event_config_wo_mapping,
        n_min_events=1,
    )
    event_cohort = EventCohort(person=person, event=event)
    return event_cohort
