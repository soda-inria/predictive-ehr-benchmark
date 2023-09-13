from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from medem.constants import (
    COLNAME_BIRTH_DATE,
    COLNAME_DEATH_DATE,
    COLNAME_FOLLOWUP_START,
    COLNAME_OUTCOME_DATETIME,
    COLNAME_PERSON,
    COLNAME_START,
)
from medem.utils import to_lazyframe, to_pandas
import polars as pl


class EventCohort:
    def __init__(
        self,
        person: pd.DataFrame = None,
        event: pd.DataFrame = None,
        folder: Path = None,
        load_inclusion_events: bool = False,
        lazy: bool = False,
    ):
        """Cohort object containing event and person dataframes.

        Parameters
        ----------
        person : pd.DataFrame, optional
            _description_, by default None
        event : pd.DataFrame, optional
            _description_, by default None
        folder : Path, optional
            Path to a local folder for writing the dataframe.
        framework: str
            Framework to use for the cohort. Can be either "pandas" (future: or "polars").
        """
        self.lazy = lazy
        if self.lazy:
            self._reader = pl.scan_parquet
        else:
            self._reader = pd.read_parquet
        if folder is not None:
            self.folder = Path(folder)

            person, event, inclusion_events = self._read_cohort(
                folder=self.folder,
                load_inclusion_events=load_inclusion_events,
                reader=self._reader,
            )
            self.name = self.folder.name

        else:
            self.folder = None
            self.name = ""
            inclusion_events = {}
        # forcing dates to datetime
        # framework_ = bd.get_backend(person)
        person_datetime_cols = [
            COLNAME_BIRTH_DATE,
            COLNAME_DEATH_DATE,
            COLNAME_FOLLOWUP_START,
            COLNAME_OUTCOME_DATETIME,
        ]
        for col in person_datetime_cols:
            if col in person.columns:
                person[col] = pd.to_datetime(person[col])
        event[COLNAME_START] = pd.to_datetime(event[COLNAME_START])
        self.event = event
        self.person = person
        self.inclusion_events = inclusion_events
        if self.lazy:
            self.event = pl.DataFrame(event).lazy()
            self.person = pl.DataFrame(person).lazy()
            self.inclusion_events = pl.DataFrame(inclusion_events).lazy()

    @staticmethod
    def _read_cohort(
        folder: str, load_inclusion_events: bool, reader
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        if not folder.exists():
            raise FileNotFoundError(f"{folder} does not exist")

        path2event = folder / "event.parquet"
        path2person = folder / "person.parquet"
        if not path2event.exists():
            raise FileNotFoundError(f"{path2event} does not exist")
        elif not path2person.exists():
            raise FileNotFoundError(f"{path2person} does not exist")

        person = reader(str(path2person))
        event = reader(str(path2event))

        inclusion_events = {}
        if load_inclusion_events:
            inclusion_names = [
                file_name
                for file_name in Path(folder).iterdir()
                if file_name.stem.startswith("inclusion")
            ]
            for inclusion_name in inclusion_names:
                inclusion_events[inclusion_name.stem] = reader(
                    str(folder / inclusion_name)
                )
        return person, event, inclusion_events

    def to_parquet(self, folder=None):
        if folder is not None:
            self.folder = Path(folder)
            self.folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing tables to {self.folder}")
        path2event = self.folder / "event.parquet"
        path2person = self.folder / "person.parquet"
        if self.lazy:
            # snappy compression needed for spark compatibility
            self.person.sink_parquet(path2person, compression="snappy")
            self.event.sink_parquet(path2event, compression="snappy")
        else:
            self.person.to_parquet(path2person)
            self.event.to_parquet(path2event)

    def __str__(self):
        if self.lazy:
            n_persons = (
                self.person.select(pl.count()).collect().to_numpy()[0][0]
            )
            n_events = self.event.select(pl.count()).collect().to_numpy()[0][0]
        else:
            n_persons = self.person.shape[0]
            n_events = self.event.shape[0]
        return f"""EventCohort {self.name}:\n {n_persons} patients\n {n_events}
                events"""

    def slice_cohort(self, person_id_subsets: List[str]):
        """
        Filter a cohort to keep only based on a list of person ids.

        Args:
            person_id_subsets (List[str]): _description_

        Returns:
            EventCohort: Subset of events as a cohort object.
        """
        if self.lazy:
            person_id_df = to_lazyframe(
                pd.DataFrame(set(person_id_subsets), columns=[COLNAME_PERSON])
            )
            sliced_cohort = EventCohort(
                person=self.person.join(
                    person_id_df, on=COLNAME_PERSON, how="inner"
                ),
                event=self.event.join(
                    person_id_df, on=COLNAME_PERSON, how="inner"
                ),
            )
        else:
            person_id_df = to_pandas(
                pd.DataFrame(set(person_id_subsets), columns=[COLNAME_PERSON])
            )
            sliced_cohort = EventCohort(
                person=self.person.merge(
                    person_id_df, on=COLNAME_PERSON, how="inner"
                ),
                event=self.event.merge(
                    person_id_df, on=COLNAME_PERSON, how="inner"
                ),
                lazy=self.lazy,
            )
        return sliced_cohort
