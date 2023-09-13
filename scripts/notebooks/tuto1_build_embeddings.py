import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from databricks import koalas as ks
from dateutil.parser import parse
from eds_scikit import improve_performances
from eds_scikit.io import HiveData
from event2vec.svd_ppmi import event2vec

from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_OUTCOME,
    COLNAME_OUTCOME_STAY_ID,
    COLNAME_PERSON,
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    COLNAME_VALUE,
    DIR2DATA,
    DIR2DOCS_COHORT,
    DIR2RESOURCES,
)
from medem.experiences.configurations import (
    CONFIG_LOS_COHORT,
    cohort_configuration_to_str,
)
from medem.experiences.pipelines import (
    OneHotEvent,
    build_vocabulary,
    restrict_to_vocabulary,
)
from medem.experiences.cohort import EventCohort
from medem.utils import save_figure_to_folders

if __name__ == "__main__":
    config = CONFIG_LOS_COHORT
    cohort_name = cohort_configuration_to_str(config)

    dir2report_imgs = DIR2DOCS_COHORT / cohort_name
    dir2report_imgs.mkdir(exist_ok=True)
    dir2cohort = DIR2DATA / cohort_name
    event_cohort = EventCohort(folder=dir2cohort)
    print(event_cohort.person.shape)
    print(event_cohort.event.shape)
    event_cohort.event.head()

    n_min_events = 10
    vocabulary = build_vocabulary(
        event=event_cohort.event, n_min_events=n_min_events
    )
    restricted_events = restrict_to_vocabulary(
        event=event_cohort.event, vocabulary=vocabulary
    )

    embeddings = event2vec(
        events=restricted_events,
        output_dir=dir2cohort,
        colname_concept=COLNAME_SOURCE_CODE,
        window_radius_in_days=30,
        window_orientation="center",
        matrix_type="numpy",
        backend="pandas",
    )
