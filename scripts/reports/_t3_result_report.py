# +
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator, FixedLocator
import textwrap
from statsmodels.nonparametric.smoothers_lowess import lowess

from medem.constants import DIR2DOCS_EXPERIENCES, DIR2EXPERIENCES, DIR2DOCS_COHORT, DIR2DATA
from medem.experiences.configurations import *
from medem.experiences.utils import ICD10_LABEL2CHAPTER
from medem.reports.efficiency_plot import plot_efficiency
from medem.reports.utils import (
    COLORMAP_FEATURIZER,
    DEMOGRAPHIC_LABELS,
    ESTIMATOR_STYLES,
    FEATURIZER_LABEL,
    METRIC_LABEL,
    MODEL_LABELS,
    XLABELS,
    annotate_icd10,
    get_experience_results,
    get_legend_handles_labels
)
from medem.utils import save_figure_to_folders

pd.set_option("display.max_columns", 200)
# -

expe_configs = [
    (
        #"timesplit__mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random_ML",
        "timesplit__mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random_hash_6954140338169467832",
        "estimator_label",
    ),
]
expe_name, col_order = expe_configs[-1]

# +
dir2expe = Path(DIR2EXPERIENCES / expe_name)
dir2results = DIR2DOCS_EXPERIENCES / (expe_name)# + "_cehr_bert")
dir2results.mkdir(exist_ok=True, parents=True)

expe_logs = get_experience_results(dir2expe, long_format=False)
expe_logs.head(3)
print(expe_logs.columns)
# -

expe_logs.head(3)

expe_logs[["n_person_subtrain", "featurizer", "estimator", "randomsearch_rs"]].value_counts()

# +
metrics = ["average_precision_score", "roc_auc_score"]
#metrics = ["train_average_precision_score", "train_roc_auc_score"]
# parameters
estimator_to_plots = [
    "ridge",
    "random_forests", 
    #CEHR_BERT_LABEL, 
    #"hist_gradient_boosting"
]
featurizer_to_plots = [
    FEATURIZER_DEMOGRAPHICS,
    FEATURIZER_COUNT,
    FEATURIZER_EVENT2VEC_TRAIN,
    FEATURIZER_SNDS,
    CEHR_BERT_LABEL,
]
featurizer_to_plots = [f_ for f_ in featurizer_to_plots if f_ in expe_logs["featurizer"].unique()]


x_name = XLABELS["n_person_subtrain"]
for metric_name in metrics:
    fig, axes = plt.subplots(1, len(estimator_to_plots), figsize=(14, 4))
    for i, est_ in enumerate(estimator_to_plots):
        mask_estimator = expe_logs["estimator"].isin([est_])
        g = sns.lineplot(
            ax=axes[i],
            data=expe_logs[mask_estimator],
            x="n_person_subtrain",
            y=metric_name,
            hue="featurizer",
            style="estimator",
            palette=COLORMAP_FEATURIZER,
            dashes=ESTIMATOR_STYLES,
            legend=False,
            #markers=True, 
            #errorbar=("se", 2),
            #err_style="bars",
            #err_kws={'capsize':10}
        )
        y_name = METRIC_LABEL[metric_name.replace("train_", "")]
        g.set(xlabel=x_name, ylabel=y_name)
        axes[i].set_title(MODEL_LABELS[est_])
        # legend
        handles, labels = get_legend_handles_labels(featurizer_to_plots=featurizer_to_plots, estimator_to_plots=estimator_to_plots)
        plt.legend(
            title="",
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1.02, 0.9),
            loc="upper left",
            borderaxespad=0,
            ncol=1,
        )
      
    save_figure_to_folders(dir2results / f"{metric_name}__performances")
# -


