from typing import List
from medem.experiences.configurations import FEATURIZER_COUNT_RANDOM_PROJ
from medem.reports.utils import (
    COLORMAP_FEATURIZER,
    DECAY_LABELS,
    DEMOGRAPHIC_LABELS,
    FEATURIZER_LABEL,
    METRIC_LABEL,
    MODEL_LABELS,
    XLABELS,
)
import seaborn as sns
import numpy as np


def plot_efficiency(
    expe_logs,
    metric_name: str = "roc_auc",
    col_order: str = "estimator_label",
    phenotype_chapter: str = None,
    estimators_list: List[str] = None,
    kind="lineplot",
    col_wrap: int = None,
):
    x_label = "n_person_subtrain"
    if phenotype_chapter is not None:
        y_name = f"c_{phenotype_chapter}__score_{metric_name}"
    else:
        y_name = metric_name

    y_label = (
        METRIC_LABEL[metric_name]
        if metric_name in METRIC_LABEL
        else metric_name
    )
    x_order = np.sort(expe_logs[x_label].unique())
    ecarted_featurizers = [FEATURIZER_COUNT_RANDOM_PROJ]
    hue_order = {
        k
        for k in COLORMAP_FEATURIZER.keys()
        if (k in expe_logs["featurizer"].unique())
        and k not in ecarted_featurizers
    }
    col = col_order
    if col_order == "estimator_label":
        if estimators_list is None:
            estimators_list = expe_logs["estimator_label"].unique()
        col_order = [
            e_label
            for e_label in list(MODEL_LABELS.values())
            if e_label in estimators_list
        ]
    elif col_order == "demographic_label":
        col_order = list(DEMOGRAPHIC_LABELS.values())
    elif col_order == "decay_label":
        col_order = list(DECAY_LABELS.values())
    g = sns.FacetGrid(
        expe_logs,
        col=col,
        col_order=col_order,
        # aspect=1.5,
        # height=4,
        col_wrap=col_wrap,
    )
    if kind == "lineplot":
        g = g.map_dataframe(
            sns.lineplot,
            x=x_label,
            y=y_name,
            hue="featurizer",
            palette=COLORMAP_FEATURIZER,
            hue_order=hue_order,
            legend=True,
        )
    elif kind == "boxplot":
        g = g.map_dataframe(
            sns.boxplot,
            x=x_label,
            order=x_order,
            y=y_name,
            hue="featurizer",
            palette=COLORMAP_FEATURIZER,
            hue_order=hue_order,
        )
    g.add_legend()
    legend_data = {
        k: g._legend_data[k]
        for k in list(COLORMAP_FEATURIZER.keys())
        if k in g._legend_data.keys()
    }
    g._legend.remove()

    g.add_legend(
        title=FEATURIZER_LABEL,
        legend_data=legend_data,
        # loc="upper center",
        bbox_to_anchor=(1.01, 0.7),
        ncol=1,
    )

    g.set_titles(template="{col_name}")
    g.set(
        xlabel=XLABELS[x_label],
        ylabel=y_label,
    )
    return g
