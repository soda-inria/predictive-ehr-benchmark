# %%
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from medem.experiences.configurations import (
    CONFIG_PROGNOSIS_COHORT,
    cohort_configuration_to_str,
)

import seaborn as sns
import matplotlib.pyplot as plt

from medem.constants import *
from pathlib import Path
from medem.experiences.configurations import *

from medem.reports.utils import (
    CEHR_BERT_LABEL_CLEAN,
    COLORMAP_FEATURIZER,
    FEATURIZER_COUNT_LABEL_CLEAN,
    FEATURIZER_LABEL,
    MODEL_LABELS,
    get_experience_results,
    TAB_COLOR,
)
from medem.utils import save_figure_to_folders


pd.set_option("display.max_columns", 150)

# %%
"""
The goal is to compare the mean combined training and evaluation time per unique chapter depending on the method.
"""

# %%
expe_name = "timesplit__icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_four_ML"
dir2expe = Path(DIR2EXPERIENCES / expe_name)
dir2results = DIR2DOCS_EXPERIENCES / expe_name
dir2results.mkdir(exist_ok=True, parents=True)
config = CONFIG_PROGNOSIS_COHORT
cohort_name = cohort_configuration_to_str(config)
dir2cohort = DIR2DATA / cohort_name
expe_logs = get_experience_results(dir2expe, long_format=False)

# %%
BERT_RESULTS = True

estimator_to_plots = {
    "ridge",
    "random_forests",
    CEHR_BERT_LABEL
    #    "hist_gradient_boosting"
}
color_palette = {
    MODEL_LABELS["ridge"]: TAB_COLOR[18],
    MODEL_LABELS["random_forests"]: TAB_COLOR[16],
    MODEL_LABELS[CEHR_BERT_LABEL]: COLORMAP_FEATURIZER[CEHR_BERT_LABEL_CLEAN],
}
featurizer_to_plots = [
    FEATURIZER_DEMOGRAPHICS,
    FEATURIZER_EVENT2VEC_TRAIN,
    # EATURIZER_CUI2VEC,
    FEATURIZER_SNDS,
    # EATURIZER_COUNT_WO_DECAY,
    FEATURIZER_COUNT_LABEL_CLEAN,
    CEHR_BERT_LABEL,
]
first_seeds = [4]
label_compute_time = "Compute time (seconds)"
results_of_interest = expe_logs.loc[
    expe_logs["estimator"].isin(estimator_to_plots)
    & expe_logs["featurizer"].isin(featurizer_to_plots)
    & (expe_logs["n_person_subtrain"] == expe_logs["n_person_subtrain"].max())
    & (expe_logs["splitting_rs"].isin(first_seeds))
]
# dividing by 21 to measure the per task mean compute time
results_of_interest[label_compute_time] = (
    pd.to_timedelta(results_of_interest["compute_time"]).dt.total_seconds() / 21
)

columns_train_test_times = ["featurizer", "estimator", label_compute_time]

# work a little bit to get back time from cehr bert results
if BERT_RESULTS:
    model_name = "CEHR_BERT_512"
    cohort_name = "icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01"
    dir2bert_logs = DIR2EXPERIENCES / (cohort_name + "_" + model_name + ".csv")
    cehr_bert_logs = pd.read_csv(dir2bert_logs)

    cehr_bert_logs["time_stamp"] = pd.to_datetime(
        cehr_bert_logs["time_stamp"], format="%m-%d-%Y-%H-%M-%S"
    )
    # the estimation is based on the logged timestamped for each run, hypothesis is that they run successively.
    successive_runs_only = cehr_bert_logs.loc[
        (cehr_bert_logs["training_percentage"] == 0.99)
        & (cehr_bert_logs["random_seed"].isin(first_seeds))
    ].sort_values("time_stamp")
    successive_runs_only[label_compute_time] = (
        successive_runs_only["time_stamp"]
        - successive_runs_only["time_stamp"].shift(1)
    ).dt.total_seconds()
    successive_runs_only["featurizer"] = CEHR_BERT_LABEL
    successive_runs_only["estimator"] = CEHR_BERT_LABEL

    train_test_times = pd.concat(
        [
            successive_runs_only[columns_train_test_times],
            results_of_interest[columns_train_test_times],
        ]
    )
else:
    train_test_times = results_of_interest[columns_train_test_times]
train_test_times["estimator"] = train_test_times["estimator"].map(
    lambda x: MODEL_LABELS[x] if x in MODEL_LABELS.keys() else x
)
train_test_times = train_test_times.rename(
    columns={"featurizer": FEATURIZER_LABEL, "estimator": "Estimator"}
)

train_test_times[FEATURIZER_LABEL] = train_test_times[FEATURIZER_LABEL].map(
    lambda x: "\n".join(CEHR_BERT_LABEL_CLEAN.split(" ")) + " (GPU)"
    if x == CEHR_BERT_LABEL
    else x + "\n(CPU)"
)
featurizer_to_plots_order = [
    "\n".join(CEHR_BERT_LABEL_CLEAN.split(" ")) + " (GPU)"
    if x == CEHR_BERT_LABEL
    else x + "\n(CPU)"
    for x in featurizer_to_plots
]

# %%
train_test_times.groupby([FEATURIZER_LABEL, "Estimator"]).agg(
    **{
        label_compute_time: pd.NamedAgg(label_compute_time, np.mean),
        f"sd {label_compute_time}": pd.NamedAgg(label_compute_time, np.std),
    }
).reset_index()


# %%
featurizer_to_plots_order

# %%
sns.set(font_scale=1, style="whitegrid")  # , context="talk")
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.grid(alpha=0.5)

g = sns.barplot(
    ax=ax,
    data=train_test_times,
    y=FEATURIZER_LABEL,
    x=label_compute_time,
    hue="Estimator",
    order=featurizer_to_plots_order,
    palette=color_palette,
)
save_figure_to_folders(dir2results / "training_testing_time_per_chapter")
ax.xaxis.set_minor_locator(
    MultipleLocator(50)
)  # must be associated w tick params
ax.tick_params(which="both", bottom=True)

# %%
