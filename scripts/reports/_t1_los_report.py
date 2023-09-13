# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from medem.constants import DIR2DOCS_EXPERIENCES, DIR2EXPERIENCES
from medem.experiences.configurations import FEATURIZER_COUNT_RANDOM_PROJ
from medem.reports.efficiency_plot import plot_efficiency
from medem.reports.utils import *
from medem.utils import save_figure_to_folders

from medem.experiences.configurations import CONFIG_LOS_COHORT

# %%

expe_configs = [
    (  # Cross-validated decay, LR and Random forest with all but local embeddings (to remove?)
        "timesplit__complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3_all_featurizers",
        "estimator_label",
        "roc_auc_score",
        0.65,
        1,
        2,
    ),
    (  # Cross-validated decay, LR and Random forest with all but local embeddings (to remove?)
        "timesplit__complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3_all_featurizers",
        "estimator_label",
        "average_precision_score",
        0.85,
        1,
        2,
    ),
    (  # Cross-validated decay, LR and Random forest with all but local embeddings (to remove?)
        "timesplit__complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3_all_featurizers",
        "estimator_label",
        "brier_score_loss",
        0,
        0.25,
        2,
    ),
]

# %%
# log bert logs
cohort_config = CONFIG_LOS_COHORT
cohort_name = cohort_configuration_to_str(cohort_config)
model_name = "CEHR_BERT_512_pipeline__target_LOS"

dir2bert_logs = DIR2EXPERIENCES / (cohort_name + "_" + model_name + ".csv")

if not dir2bert_logs.exists():
    cehr_bert_result = (
        get_cehr_bert_results(
            cohort_config=cohort_config, model_name=model_name
        )
        .rename(
            columns={
                "roc_auc": "roc_auc_score",
                "pr_auc": "average_precision_score",
                # "bier_score_loss":"brier_score_loss"
            }
        )
        .drop(columns=["brier_score_loss"])
        .rename(columns={"bier_score_loss": "brier_score_loss"})
    )
    cehr_bert_result.to_csv(dir2bert_logs)
else:
    cehr_bert_result = pd.read_csv(dir2bert_logs)
cehr_bert_result["featurizer"] = CEHR_BERT_LABEL_CLEAN
# %%
sns.set(font_scale=1, style="whitegrid", rc={"figure.figsize": (7, 4.5)})

for expe_name, col_order, metric_name, y_inf, y_sup, col_wrap in expe_configs:
    dir2expe = Path(DIR2EXPERIENCES / expe_name)
    expe_logs = get_experience_results(dir2expe, long_format=False)
    # expe_logs["n_person_subtrain"] = expe_logs["n_person_subtrain"].astype(int)
    # convert percent training into n_samples
    n_train_samples = expe_logs["n_person_subtrain"].max()
    cehr_bert_result["n_person_subtrain"] = (
        cehr_bert_result["training_percentage"] * n_train_samples
    ).astype(
        int
    )  # .astype(str)
    cehr_bert_result

    g = plot_efficiency(
        # kind="boxplot",
        expe_logs=expe_logs,
        metric_name=metric_name,
        col_order=col_order,
        col_wrap=col_wrap,
        estimators_list=["Logistic regression", "Random Forest"],
    )
    # #add cehr-bert results
    for ax in g.axes:
        sns.lineplot(
            ax=ax,
            data=cehr_bert_result.loc[~cehr_bert_result[metric_name].isna()],
            x="n_person_subtrain",
            y=metric_name,
            hue="featurizer",
            palette=COLORMAP_FEATURIZER,
            legend=False,
            errorbar=("se", 2),
        )
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.xaxis.set_major_formatter("{x:.0f}")
        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(
            MultipleLocator(2500)
        )  # must be associated w tick params
        ax.tick_params(which="both", bottom=True)
        ax.grid(alpha=0.5)
    g.set(
        ylim=(y_inf, y_sup),  # xlim=(0, n_train_samples),
        yticks=np.arange(y_inf, y_sup + 0.05, 0.05),
    )
    # add cehr-bert to legend
    legend_data = {
        k.replace(" ", "\n"): g._legend_data[k]
        for k in list(COLORMAP_FEATURIZER.keys())
        if k in g._legend_data.keys()
    }
    legend_data[CEHR_BERT_LABEL_CLEAN.replace(" ", "\n")] = Line2D(
        [0], [0], color=COLORMAP_FEATURIZER[CEHR_BERT_LABEL_CLEAN], lw=2
    )
    g._legend.remove()
    g.add_legend(
        title=FEATURIZER_LABEL,
        legend_data=legend_data,
        loc="upper left",
        bbox_to_anchor=(0.1, 1.15),
        prop={"size": 10},
        borderaxespad=0,
        ncol=5,
    )
    save_figure_to_folders(
        Path(f"experiences") / expe_name / f"{metric_name}_performances",
        to_paper_dir=True,
    )
    # plt.plot()
# %%
# # Decay study

# what decays are better ? No decay and 1 day decay (approximately same results for AUPRC or ridge)
metric = "roc_auc_score"
sns.boxplot(
    expe_logs.loc[expe_logs["estimator"] == "random_forests"],
    x="decay_label",
    y=metric,
    hue="featurizer",
)
plt.xticks(rotation=45)
