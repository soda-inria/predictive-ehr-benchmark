import numpy as np
import pandas as pd
import pytest

from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_OUTCOME,
    COLNAME_PERSON,
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    COLNAME_START,
)
from medem.experiences.configurations import (
    PATH2SNDS_EMBEDDINGS,
)
from dateutil.parser import parse

from medem.experiences.utils import (
    AFTERNOON,
    MORNING,
    NIGHT,
    event_train_test_split,
    get_scores,
    get_prognosis_prevalence,
    get_time_of_day,
)

expected_one_hot = np.array([[4.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
expected_decay = np.hstack(
    [
        expected_one_hot,
        np.array([[0.00673795, 0.0, 0.04978707], [0.0, 0.09957414, 0.0]]),
    ]
)


new_event = pd.DataFrame(
    {
        COLNAME_PERSON: [5, 5, 5, 6, 7, 7, 7],
        COLNAME_SOURCE_CODE: [
            "A04AA01",
            "A04AA01",
            "GFEA004",
            "OOV001",
            "48891-6",
            "48891-6",
            "48891-6",
        ],
        COLNAME_START: [
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-05 00:00:00",
        ],
    },
)
new_event[COLNAME_START] = pd.to_datetime(new_event[COLNAME_START])
new_person = pd.DataFrame(
    {
        COLNAME_PERSON: [7, 6, 5],
        COLNAME_OUTCOME: [0, 1, 1],
        COLNAME_FOLLOWUP_START: [
            "2021-01-07 00:00:00",
            "2021-01-05 00:00:00",
            "2021-01-07 00:00:00",
        ],
    }
)
new_person[COLNAME_FOLLOWUP_START] = pd.to_datetime(
    new_person[COLNAME_FOLLOWUP_START]
)


@pytest.mark.parametrize(
    "train_size, test_size, random_seed, expected_event_train_size, expected_person_train_size",
    [
        (0.5, None, 0, 5, 1),
        (None, 0.5, 1, 2, 1),
        (1, None, 0, 7, 2),
    ],
)
def test_event_train_test_split(
    mock_event_cohort,
    train_size,
    test_size,
    random_seed,
    expected_event_train_size,
    expected_person_train_size,
):
    (
        train_event,
        test_event,
        train_person,
        test_person,
    ) = event_train_test_split(
        event_cohort=mock_event_cohort,
        train_size=train_size,
        test_size=test_size,
        random_seed=random_seed,
    )
    assert len(train_event) == expected_event_train_size
    assert (
        len(test_event)
        == len(mock_event_cohort.event) - expected_event_train_size
    )
    assert len(train_person) == expected_person_train_size
    assert (
        len(test_person)
        == len(mock_event_cohort.person) - expected_person_train_size
    )


binary_true_y = np.array([0, 1, 1, 0, 1])
binary_y_prob = np.array(
    [
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
    ]
)
expected_metrics = [
    "brier_score_loss",
    "roc_auc_score",
    "accuracy_score",
    "average_precision_score",
    "tn",
    "fp",
    "fn",
    "tp",
]
classes = ["0", "1", "2", "3", "4"]
multilabel_true_y = np.repeat(
    binary_true_y.reshape((-1, 1)), repeats=len(classes), axis=1
)
multilabel_y_prob = [binary_y_prob] * len(classes)
expected_multilabel_metrics = [
    f"c_{c}__score_{metric}" for c in classes for metric in expected_metrics
]


@pytest.mark.parametrize(
    "classes, expected_metrics, y_true, y_prob",
    [
        (
            classes,
            expected_multilabel_metrics,
            multilabel_true_y,
            multilabel_y_prob,
        ),
        (None, expected_metrics, binary_true_y, binary_y_prob),
    ],
)
def test_get_scores(classes, expected_metrics, y_true, y_prob):
    scores = get_scores(y_true=y_true, y_prob=y_prob, classes=classes)

    np.testing.assert_array_equal(list(scores.keys()), expected_metrics)


def test_describe_prognosis():
    y = pd.DataFrame({"y": [["2"], ["11"], ["9", "11", "2"]]})["y"]
    prognosis_description = get_prognosis_prevalence(y=y, cim10_nb_digits=1)
    np.testing.assert_array_equal(
        prognosis_description.columns.values,
        np.array(
            [
                "Neoplasms",
                "Diseases of the digestive system",
                "Diseases of the circulatory system",
            ],
        ),
    )


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        (parse("2020-01-01 07:00:00"), MORNING),
        (parse("2020-01-01 12:00:00"), AFTERNOON),
        (parse("2020-01-01 22:00:00"), NIGHT),
        (parse("2020-01-01 05:00:00"), NIGHT),
    ],
)
def test_get_time_of_day(timestamp, expected):
    assert get_time_of_day(timestamp) == expected
