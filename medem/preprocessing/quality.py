from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import LogNorm

from medem.utils import force_datetime, to_lazyframe
import polars as pl


def time_plot(
    df: pd.DataFrame,
    colname_datetime: str = "start",
    ax: plt.axes = None,
    label: str = None,
) -> Tuple[pd.DataFrame, plt.axes]:
    """
    Return a timeplot with year-month granularity

    Parameters
    ----------
    df : Union[psql.DataFrame, ks.DataFrame]
        Raw I2B2 dataframe

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df_ = force_datetime(df, colnames_datetime=colname_datetime)
    df_["year"] = df_[colname_datetime].dt.year
    df_["month"] = df_[colname_datetime].dt.month
    df_year_month = df_.groupby(["year", "month"])[colname_datetime].count()

    df_year_month = df_year_month.reset_index().sort_values(["year", "month"])
    df_year_month.columns = ["year", "month", "count"]
    df_year_month["day"] = 1
    # Catch invalid years/months
    min_year = 1850
    max_year = 2030
    absurd_years = (df_year_month["year"] <= min_year) | (
        df_year_month["year"] >= max_year
    )
    n_absurd_years = df_year_month.loc[absurd_years, "count"].values.sum()
    if n_absurd_years > 0:
        logger.warning(
            f"Got {n_absurd_years} aberrant years values:\n {df_year_month.loc[absurd_years, ['year', 'month', 'count']].values}"
        )

    absurd_months = ~df_year_month["month"].isin(np.arange(1, 13))
    n_absurd_months = df_year_month.loc[absurd_months, "count"].values.sum()
    if n_absurd_months > 0:
        logger.warning(
            f"Got {n_absurd_months} aberrant month values:\n {df_year_month.loc[absurd_months, ['year', 'month', 'count']].values}"
        )
    df_year_month = df_year_month.loc[~(absurd_months | absurd_years)]
    df_year_month["year-month"] = pd.to_datetime(
        df_year_month[["year", "month", "day"]]
    )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=[14, 6])
    if label is None:
        label = "count"
    df_year_month.set_index("year-month").plot(
        y="count", ax=ax, legend=False, label=label
    )
    return df_year_month, ax


def temporal_heatmap(
    event: pd.DataFrame,
    colname_of_interest: str,
    n_top_counts: int = 1000,
    plot_heatmap: bool = True,
    count_norm: str = "log",
    colname_datetime: str = "start",
    colname_stay_id: str = "visit_occurrence_id",
) -> Tuple[pd.DataFrame, plt.axes]:
    """Aggregate event on two axis to produce a count matrix of events with :
    - on the x axis : time (by year as default)
    - on the y axis : a dimension of the data of interest, eg. diagnostic codes or care functional units (UFR).

    Parameters
    ----------
    event : pd.DataFrame
        Event DataFrame from :class:`~deepacau.data_models.events.EventsModel`.
    concept_colname : str
        The dimension of interest in the data, eg. diagnostic codes or care functional units (UFR).
    n_top_counts: int
        Only output the `n_top_counts` in the count matrix, by default 1000.
    plot_heatmap : bool, optional
        Plot the heatmap, by default True.
    count_norm: str
        Color normalization for the heatmap. Support "log" or "linear", by default "log".
    temporal_grid_level : str, optional
        Time granularity, by default year.
    colname_datetime: str
        Column name used for the time axis.
    colname_stay_id: str
        Column name used for the counting of event.

    Returns
    -------
    Tuple[pd.DataFrame, plt.axes]
        _description_
    """
    if colname_of_interest not in event.columns:
        raise ValueError(
            f"concept_colname should be in {event.columns}, got {colname_of_interest}"
        )
    event_ = to_lazyframe(event)
    #
    grouped_data = (
        (
            event_.groupby(
                [
                    pl.col(colname_datetime).dt.year(),
                    pl.col(colname_of_interest),
                ]
            ).agg(pl.count(colname_stay_id))
        )
        .collect()
        .to_pandas()
        .reset_index()
        .rename(columns={colname_stay_id: "n_events"})
    )

    grouped_data_ = grouped_data.copy()
    time_concept_matrix = grouped_data_.pivot(
        index=colname_datetime,
        columns=colname_of_interest,
        values="n_events",
    ).fillna(0)

    concept_counts = pd.DataFrame(
        time_concept_matrix.sum(axis=0), columns=["count"]
    ).sort_values("count", ascending=False)
    top_concepts = concept_counts[:n_top_counts].index

    time_top_concept_matrix = time_concept_matrix.loc[
        :, top_concepts
    ].transpose()
    if plot_heatmap:
        _, ax = plt.subplots(1, 1, figsize=(14, 14))
        if count_norm == "log":
            norm = LogNorm()
        else:
            norm = None
        sns.heatmap(time_top_concept_matrix, norm=norm)
    return time_top_concept_matrix, ax
