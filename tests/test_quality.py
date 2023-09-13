from datetime import datetime

import numpy as np
import pandas as pd


def test_time_plot(mock_event_cohort):
    from medem.preprocessing.quality import time_plot

    mock_event = mock_event_cohort.event
    abberant_row = mock_event.iloc[0]
    time_colname = "start"
    abberant_row[time_colname] = datetime.strptime("1789-07-14", "%Y-%m-%d")
    test_event_ = pd.concat(
        [mock_event, pd.DataFrame(abberant_row).transpose()], axis=0
    )
    test_event_[time_colname] = pd.to_datetime(test_event_[time_colname])
    event_year_month, _ = time_plot(test_event_, colname_datetime=time_colname)

    test_year = 2018
    test_month = 1

    mock_event[time_colname] = pd.to_datetime(mock_event[time_colname])
    n_events_expected = np.sum(
        (mock_event[time_colname].dt.year == test_year)
        & (mock_event[time_colname].dt.month == test_month)
    )
    n_events_test_year_month = event_year_month.loc[
        (event_year_month["year"] == test_year)
        & (event_year_month["month"] == test_month),
        "count",
    ]
    if len(n_events_test_year_month) > 0:
        n_events_test_year_month = n_events_test_year_month.values[0]
    else:
        n_events_test_year_month = 0
    assert event_year_month.shape[1] == 5
    assert n_events_expected == n_events_test_year_month


def test_temporal_heatmap(mock_event_cohort):
    from medem.preprocessing.quality import temporal_heatmap

    time_top_concept_matrix, _ = temporal_heatmap(
        event=mock_event_cohort.event,
        colname_of_interest="event_source_concept_id",
        plot_heatmap=True,
    )

    assert time_top_concept_matrix.loc["48891-6", 2019] == 2
