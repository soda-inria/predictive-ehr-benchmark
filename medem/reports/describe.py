"""Scripts to summarize a cohort of patients.
"""


from pathlib import Path
import pandas as pd
from medem.experiences.cohort import EventCohort
from medem.experiences.utils import get_prognosis_prevalence
from medem.utils import add_age


def describe_cohort(dir2cohort: str, target_type: str = "binary"):
    """Describe a cohort of patients.

    Parameters
    ----------
    dir2cohort : str
        Path to the directory containing the cohort.
    target_type : str, optional
        Type of target, by default "binary". Also supports "multilabel".
    """
    event_cohort = EventCohort(folder=dir2cohort)

    targets = add_age(event_cohort.person, "inclusion_event_start")

    targets["age_year"] = targets["age_in_days"] / 365
    events = event_cohort.event
    cohort_description = {}
    # targets
    N = len(targets)
    cohort_description["N"] = N
    n_female = len(targets[targets["gender_source_value"] == "f"])
    cohort_description["Median age"] = targets["age_year"].median()
    cohort_description["Female"] = 100 * n_female / N
    if target_type == "binary":
        prevalences = targets["y"].value_counts() / N
        prevalences = ", ".join(
            [
                f"{t}: {100*v:.2f}"
                for t, v in zip(prevalences.index, prevalences.values)
            ]
        )
    else:
        prevalences = get_prognosis_prevalence(targets["y"])
    cohort_description["prevalences"] = prevalences
    # events
    N_events = len(events)
    cohort_description["n_events"] = N_events
    event_types = events["event_source_type_concept_id"].value_counts()
    cohort_description["n_events_by_type"] = ", ".join(
        [
            f"{t}: {int(v/1000)}K"
            for t, v in zip(event_types.index, event_types.values)
        ]
    )

    cohort_summary = pd.DataFrame.from_dict(
        cohort_description, orient="index", columns=[Path(dir2cohort).name]
    )
    return cohort_summary
