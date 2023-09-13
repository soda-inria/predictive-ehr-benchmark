# %%
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator
import textwrap
from statsmodels.nonparametric.smoothers_lowess import lowess

from medem.constants import (
    DIR2DOCS_EXPERIENCES,
    DIR2EXPERIENCES,
    DIR2DOCS_COHORT,
    DIR2DATA,
    COLNAME_OUTCOME,
)
from medem.experiences.configurations import *
from medem.reports.efficiency_plot import plot_efficiency
from medem.experiences.cohort import EventCohort

from medem.reports.utils import (
    CEHR_BERT_LABEL_CLEAN,
    COLORMAP_FEATURIZER,
    DEMOGRAPHIC_LABELS,
    ESTIMATOR_STYLES,
    FEATURIZER_COUNT_LABEL_CLEAN,
    FEATURIZER_LABEL,
    METRIC_LABEL,
    MODEL_LABELS,
    XLABELS,
    annotate_icd10,
    get_experience_results,
    get_legend_handles_labels,
    get_cehr_bert_results_prognosis,
    ICD10_LABEL2CHAPTER,
)
from medem.experiences.utils import get_prognosis_prevalence
from medem.utils import save_figure_to_folders

pd.set_option("display.max_columns", 150)
# -

expe_configs = [
    (
        "timesplit__icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_four_ML",
        # "timesplit__icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_hash_7406683001431224967", # test with hgb
        "estimator_label",
    )
]
expe_name, col_order = expe_configs[-1]

# %%
config = CONFIG_PROGNOSIS_COHORT
cohort_name = cohort_configuration_to_str(config)
config = CONFIG_PROGNOSIS_COHORT
cohort_name = cohort_configuration_to_str(config)

dir2report_imgs = DIR2DOCS_EXPERIENCES / expe_name
dir2report_imgs.mkdir(exist_ok=True)
dir2cohort = DIR2DATA / cohort_name
print(dir2cohort)
print(dir2report_imgs)
# -
"""# rm doublon for some expes
from datetime import datetime
cehr_bert_logs = get_cehr_bert_results_prognosis(config, model_name, test_prevalences=test_prevalences)
print(cehr_bert_logs["time_stamp"].map(lambda x: datetime.strptime(x, "%m-%d-%Y-%H-%M-%S")).dt.day.value_counts())

dir2bert_results = dir2cohort / "evaluation_train_val_split"
model_name = "CEHR_BERT_512"

all_metrics = []
for target in range(1,23):
    target_metric_dir = dir2bert_results/f"{model_name}_pr2_pipeline__target_{target}"/"metrics"
    if target_metric_dir.exists():
        target_metric_files = list(target_metric_dir.iterdir())
        for f in target_metric_files:
            if f.name.startswith("08-19"):
                f.unlink()
"""

# %%
BERT_RESULTS = True

dir2expe = Path(DIR2EXPERIENCES / expe_name)
dir2results = dir2report_imgs
dir2results.mkdir(exist_ok=True, parents=True)

expe_logs = get_experience_results(dir2expe, long_format=True)
# prevalences missing

if BERT_RESULTS:
    model_name = "CEHR_BERT_512"
    dir2bert_logs = DIR2EXPERIENCES / (cohort_name + "_" + model_name + ".csv")
    force_bert_result_collection = False
    if (not dir2bert_logs.exists()) or force_bert_result_collection:
        event_cohort = EventCohort(folder=dir2cohort)
        person_w_split = event_cohort.person.merge(
            pd.read_parquet(dir2cohort / "dataset_split.parquet")
        )
        test_person = person_w_split.loc[
            person_w_split["dataset"] == "external_test"
        ]
        test_prevalences = (
            get_prognosis_prevalence(test_person[COLNAME_OUTCOME])
            .transpose()
            .merge(
                pd.DataFrame.from_dict(
                    ICD10_LABEL2CHAPTER, orient="index", columns=["target"]
                ),
                left_index=True,
                right_index=True,
            )
        )
        cehr_bert_logs = get_cehr_bert_results_prognosis(
            config, model_name, test_prevalences=test_prevalences
        )
        cehr_bert_logs.to_csv(dir2bert_logs)
    else:
        cehr_bert_logs = pd.read_csv(dir2bert_logs)
    # add total number of person
    n_train_samples = expe_logs["n_person_subtrain"].max()
    cehr_bert_logs["n_person_subtrain"] = (
        cehr_bert_logs["training_percentage"] * n_train_samples
    ).astype(int)
    cehr_bert_logs["featurizer"] = CEHR_BERT_LABEL_CLEAN
    cehr_bert_logs = cehr_bert_logs.rename(
        columns={
            "pr_auc": "average_precision_score",
            "roc_auc": "roc_auc_score",
            "random_seed": "splitting_rs",
        }
    ).drop(columns=["target"])
    cehr_bert_logs["target_label"] = cehr_bert_logs["target_label"].astype(str)

    test_prevalences = (
        cehr_bert_logs[["icd10 chapter", "prevalence", "target_label"]]
        .sort_values("prevalence", ascending=False)
        .drop_duplicates()
    )

    expe_logs_w_prevalences = expe_logs.merge(
        test_prevalences, on="target_label", how="left"
    )
    all_results = pd.concat([expe_logs_w_prevalences, cehr_bert_logs], axis=0)
else:
    pass
    # test_prevalences = test_prevalences.reset_index(
    #     names=["icd10 chapter"]
    # ).rename(columns={"target": "target_label"})
    # all_results = expe_logs.merge(test_prevalences, on="target_label")
# %%

cehr_bert_logs[
    ["training_percentage", "splitting_rs"]
].value_counts().to_frame().sort_values(["training_percentage", "splitting_rs"])

# %%
# parameters
sns.set(font_scale=1, style="whitegrid")  # , context="talk")

estimator_to_plots = [
    "ridge",
    "random_forests",
    # "hist_gradient_boosting",
    CEHR_BERT_LABEL,
]
featurizer_to_plots = [
    FEATURIZER_EVENT2VEC_TRAIN,
    FEATURIZER_SNDS,
    CEHR_BERT_LABEL_CLEAN,
    FEATURIZER_COUNT_LABEL_CLEAN,
    FEATURIZER_DEMOGRAPHICS,
]

x_name = XLABELS["n_person_subtrain"]

naive_results = all_results.loc[all_results["estimator"] == "naive_baseline"]

max_person = all_results["n_person_subtrain"].max()
metrics_to_plot = ["roc_auc_score", "average_precision_score"]
chapters_to_plot = [
    "macro",
    "weighted",
    *test_prevalences["target_label"].values,
]
for metric_name in metrics_to_plot:
    for chapter_id in chapters_to_plot:
        mask_estimators = (
            all_results["estimator"].isin(estimator_to_plots)
        ) & (all_results["featurizer"].isin(featurizer_to_plots))
        if chapter_id == "macro":
            chapter_name_ = "Macro average of the 21 ICD10 chapters"
            data_to_plot = all_results.loc[mask_estimators]
            y_name = "Macro " + METRIC_LABEL[metric_name]
            # prepare naive estimator baseline
            naive_metric = naive_results[metric_name].mean()
        elif chapter_id == "weighted":
            y_name = (
                METRIC_LABEL[metric_name]
                + " across ICD10 chapters, \nweighted by prevalence"
            )
            chapter_name_ = y_name

            prevalences_normalization = test_prevalences[
                "prevalence"
            ].sum() / len(test_prevalences)
            data_to_plot = all_results.loc[mask_estimators]
            data_to_plot[metric_name] = (
                data_to_plot[metric_name] * data_to_plot["prevalence"]
            ) / prevalences_normalization
            data_to_plot = (
                data_to_plot.groupby(
                    [
                        "splitting_rs",
                        "estimator",
                        "featurizer",
                        "n_person_subtrain",
                    ]
                )
                .agg(**{metric_name: pd.NamedAgg(metric_name, "mean")})
                .reset_index()
            )

            data_to_plot["line_label"] = data_to_plot.apply(
                lambda x: x["featurizer"] + " + " + MODEL_LABELS[x["estimator"]]
                if x["estimator"] != CEHR_BERT_LABEL
                else x["featurizer"],
                axis=1,
            )

            #
            naive_metric = (
                naive_results[metric_name]
                * naive_results["prevalence"]
                / prevalences_normalization
            ).mean()

        else:
            y_name = METRIC_LABEL[metric_name]

            mask_chapter_id = test_prevalences["target_label"] == chapter_id
            chapter_name_ = test_prevalences.loc[mask_chapter_id][
                "icd10 chapter"
            ].values[0]
            chapter_test_prevalence_ = test_prevalences.loc[mask_chapter_id][
                "prevalence"
            ].values[0]
            data_to_plot = all_results.loc[
                mask_estimators & (all_results["target_label"] == chapter_id),
            ]
            naive_metric = naive_results.loc[
                naive_results["target_label"] == chapter_id, metric_name
            ].values[0]

        logger.info(f"Making figure for chapter { chapter_name_}")

        fig_width = 8
        if (chapter_id == "weighted") and (
            metric_name == "average_precision_score"
        ):
            fig_width = 6
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4))
        mask_ridge = data_to_plot["estimator"].isin(["ridge"])
        g = sns.lineplot(
            data=data_to_plot[mask_ridge],
            x="n_person_subtrain",
            y=metric_name,
            hue="featurizer",
            style="estimator",
            palette=COLORMAP_FEATURIZER,
            dashes=ESTIMATOR_STYLES,
            legend=False,
            # markers=True,
            errorbar=("se", 2),
            err_style="bars",
            err_kws={"capsize": 10},
        )
        g = sns.lineplot(
            data=data_to_plot[~mask_ridge],
            x="n_person_subtrain",
            y=metric_name,
            hue="featurizer",
            style="estimator",
            palette=COLORMAP_FEATURIZER,
            dashes=ESTIMATOR_STYLES,
            legend=False,
            # markers=True,
            errorbar=("se", 2),
            # err_kws={'capsize':10}
        )
        g.set(xlabel=x_name, ylabel=y_name)
        ax.grid(alpha=0.5)
        # legend
        handles, labels = get_legend_handles_labels(
            featurizer_to_plots=featurizer_to_plots,
            estimator_to_plots=estimator_to_plots,
        )
        if (chapter_id == "weighted") and (
            metric_name == "average_precision_score"
        ):
            # complicated code to put labels next to lines (from https://python-graph-gallery.com/web-line-chart-with-labels-at-line-end/)
            LABEL_LINE_LABEL = [
                0.42,  # cbert
                0.57,  # forest count
                0.52,  # forest demographics
                0.5,  # forest local embeddings
                0.545,  # forest snds
                0.45,  # ridge count
                0.475,  # ridge demographics
                0.35,  # ridge local embeddings
                0.39,  # ridge snds
            ]

            PAD_RIGHT = 700
            x_end = max_person + PAD_RIGHT
            PAD = 100
            for idx, group in enumerate(data_to_plot["line_label"].unique()):
                label_data = data_to_plot.loc[
                    data_to_plot["line_label"] == group
                ]
                x_start = label_data["n_person_subtrain"].max()
                label_data = label_data.loc[
                    label_data["n_person_subtrain"] == x_start
                ]
                color = COLORMAP_FEATURIZER[label_data["featurizer"].values[0]]
                text = label_data["line_label"].values[0]
                y_start = label_data[metric_name].values[0]
                y_end = LABEL_LINE_LABEL[idx]
                ax.plot(
                    [x_start, (x_start + x_end - PAD) / 2, x_end - PAD],
                    [y_start, y_end, y_end],
                    color=color,
                    alpha=0.5,
                    # ls="dashed"
                )

                # Add country text
                ax.text(
                    x_end,
                    y_end,
                    text,
                    color=color,
                    fontsize=10,
                    weight="bold",
                    va="center",
                )
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.set_xlim(0, max_person + PAD_RIGHT)

                ax.xaxis.set_major_locator(MultipleLocator(2000))
                ax.xaxis.set_major_formatter("{x:.0f}")
                # For the minor ticks, use no labels; default NullFormatter.
                ax.xaxis.set_minor_locator(MultipleLocator(1000))  # must be associated w tick params

                ax.tick_params(which="both", bottom=True)

                # ax.set_ylim(-4.1, 3)
        else:
            plt.legend(
                title="",
                handles=handles,
                labels=labels,
                bbox_to_anchor=(1.02, 0.65),
                loc="upper left",
                prop={"size": 11},
                borderaxespad=0,
                ncol=1,
            )
        # nave line and label
        ax.axhline(
            y=naive_metric,
            color="black",
            linestyle="-",
            xmin=0,
            xmax=0.98,
        )

        ha = "right"
        x_naive = max_person
        va = "bottom"
        g.text(
            s="Previous stay baseline",
            x=x_naive,
            y=naive_metric,
            color="black",
            va=va,
            ha=ha,
        )
        # annotate
        chapter_wrap = "\n".join(
            textwrap.wrap(chapter_name_, 25, break_long_words=False)
        )
        if not chapter_id in ["macro", "weighted"]:
            chapter_txt = (
                f"ICD10 chapter {chapter_id}:\n"
                + chapter_wrap
                + f"\n\nTest prevalence: {chapter_test_prevalence_:.1f}%"
            )
        else:
            chapter_txt = chapter_wrap
        if chapter_id != "weighted":
            ax = plt.gca()
            ax.text(
                1.02,
                1.0,
                chapter_txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontweight="bold",
            )
        # plt.show()
        save_figure_to_folders(dir2results / f"{metric_name}__c_{chapter_id}")
# -
# # Prevalence plot

# %%
mask_all_training_samples = (
    all_results["n_person_subtrain"] == np.max(all_results["n_person_subtrain"])
) | (
    all_results["n_person_subtrain"]
    == np.max(cehr_bert_logs["n_person_subtrain"])
)
mask_estimators = all_results["estimator"].isin(
    ["ridge", "random_forests", CEHR_BERT_LABEL]
)
featurizers_to_plot = [
    FEATURIZER_COUNT,
    FEATURIZER_COUNT_WO_DECAY,
    FEATURIZER_EVENT2VEC_TRAIN,
    FEATURIZER_SNDS,
    CEHR_BERT_LABEL,
]
mask_featurizers = all_results["featurizer"].isin(featurizer_to_plots)

prevalence_results = all_results.loc[
    mask_all_training_samples & mask_estimators & mask_featurizers
]
prevalences = prevalence_results[["prevalence"]]
prevalences.describe(percentiles=[0.1, 0.33, 0.5, 0.66, 0.75, 0.9])
bins = [0, 1, 4, 6, 11, 15, 20, 30, 50, 100]
bin_labels = [
    f"{int(l_b)}-{int(u_b)}" for (l_b, u_b) in zip(bins[:-1], bins[1:])
]
prevalence_bin_label = "Prevalence (binned)"
prevalence_results[prevalence_bin_label] = pd.cut(
    prevalence_results["prevalence"], bins=bins, labels=bin_labels
).astype(str)
prevalence_results.sort_values(prevalence_bin_label, inplace=True)
# %%
# prevalence plot
xscale = "log"

mask_cehr_bert = prevalence_results["estimator"] == CEHR_BERT_LABEL
other_estimator_results = prevalence_results.loc[~mask_cehr_bert]
other_estimator_results.loc[:, "estimator_group"] = other_estimator_results[
    "estimator"
].apply(lambda x: MODEL_LABELS[x])
# duplicate the cehrt bert data to plot on each featurizer
if BERT_RESULTS:
    cehr_bert_results_r = prevalence_results[mask_cehr_bert]
    cehr_bert_results_r.loc[:, "estimator_group"] = MODEL_LABELS["ridge"]
    cehr_bert_results_rf = prevalence_results[mask_cehr_bert]
    cehr_bert_results_rf.loc[:, "estimator_group"] = MODEL_LABELS[
        "random_forests"
    ]
    prevalence_results_lplot = pd.concat(
        [other_estimator_results, cehr_bert_results_r, cehr_bert_results_rf]
    )
else:
    prevalence_results_lplot = other_estimator_results
handles, labels = get_legend_handles_labels(
    featurizer_to_plots=featurizer_to_plots,
    estimator_to_plots=estimator_to_plots,
)

plot_type = "boxplot"
estimators = ["ridge", "random_forests"]
for metric_name in ["roc_auc_score", "average_precision_score"]:
    for i, estimator_group in enumerate(estimators):
        prevalence_results_lplot_by_estimator = prevalence_results_lplot.loc[
            prevalence_results_lplot["estimator_group"]
            == MODEL_LABELS[estimator_group]
        ]
        prevalence_results_lplot_by_estimator["prevalence"] = np.round(
            prevalence_results_lplot_by_estimator["prevalence"], 4
        )
        # g = sns.FacetGrid(data=prevalence_results_lplot_by_estimator, aspect=1.8, height=4)
        sns.set_style("whitegrid")
        if plot_type == "scatter":
            fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
            sns.scatterplot(
                data=prevalence_results_lplot_by_estimator,
                x="prevalence",
                y=metric_name,
                hue="featurizer",
                palette=COLORMAP_FEATURIZER,
                legend=False,
                ax=ax,
            )
            # add lowess
            for featurizer_ in prevalence_results_lplot_by_estimator[
                "featurizer"
            ].unique():
                featurizer_data_to_plot = (
                    prevalence_results_lplot_by_estimator.loc[
                        prevalence_results_lplot_by_estimator["featurizer"]
                        == featurizer_
                    ]
                )
                xx = featurizer_data_to_plot["prevalence"]
                yy = featurizer_data_to_plot[metric_name]
                xy_pred_lowess = lowess(
                    yy,
                    xx,
                    **{"frac": 0.9, "it": 3},
                ).T
                ax.plot(
                    xy_pred_lowess[0],
                    xy_pred_lowess[1],
                    color=COLORMAP_FEATURIZER[featurizer_],
                )
            if xscale == "log":
                xticks_locators = [1, 2, 5, 10, 20, 50]
                ax.set(xscale=xscale)
                ax.set_xticks(xticks_locators, xticks_locators)

            if i == 1:
                ax.legend(
                    title="Featurizers + " + MODEL_LABELS[estimator_group],
                    handles=handles[1:-3],
                    labels=labels[1:-3],
                    prop={"size": 7.5},
                    loc="lower right",
                    borderaxespad=0,
                    ncol=2,
                )
            ax.set_title(MODEL_LABELS[estimator_group])
        elif plot_type == "boxplot":
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(7, 3.5),
                gridspec_kw={"height_ratios": [1, 4]},
                sharex=True,
            )
            plt.subplots_adjust(wspace=0, hspace=0.2)
            ax_count = axes[0]
            n_prevalence_by_bins = (
                prevalence_results_lplot_by_estimator[
                    [prevalence_bin_label, "target_label"]
                ]
                .drop_duplicates()
                .rename(columns={"target_label": "Number of\nchapters"})
                .groupby(prevalence_bin_label)
                .count()
                .reset_index()
            )
            # for
            sns.barplot(
                ax=ax_count,
                x=prevalence_bin_label,
                y="Number of\nchapters",
                data=n_prevalence_by_bins,
                color="grey",
                order=[
                    l_
                    for l_ in bin_labels
                    if l_ in n_prevalence_by_bins[prevalence_bin_label].unique()
                ],
            )
            ax_count.set(xlabel="", yticks=[0, 3])
            ax_count.yaxis.set_minor_locator(FixedLocator([0, 1, 2, 3]))

            ax = axes[1]
            sns.pointplot(
                data=prevalence_results_lplot_by_estimator,
                x=prevalence_bin_label,
                y=metric_name,
                hue="featurizer",
                errorbar=("ci", 95),
                join=True,
                order=[
                    l_
                    for l_ in bin_labels
                    if l_
                    in prevalence_results_lplot_by_estimator[
                        prevalence_bin_label
                    ].unique()
                ],
                palette=COLORMAP_FEATURIZER,
                # medianprops={"color": "black"},
                # notch=False,
                # boxprops=dict(alpha=0.8),
                dodge=0.5,
                ax=ax,
                orient="v",
                capsize=0.2,
            )
            ax.legend_.remove()
            labels_ = [
                l + "\n+ " + MODEL_LABELS[estimator_group]
                if l != CEHR_BERT_LABEL_CLEAN
                else l
                for l in labels[1:-3]
            ]
            fig.legend(
                title="Pipelines",
                handles=handles[1:-3],
                labels=labels_,
                bbox_to_anchor=(0.91, 0.5),
                prop={"size": 10},
                loc="center left",
                # loc="lower right",
                borderaxespad=0,
                ncol=1,
            )
        if metric_name == "roc_auc_score":
            ax.set(ylim=(0.5, 1))
        else:
            ax.set(ylim=(0.15, 0.8))
        x_name = "ICD10 chapter prevalences (%)"
        y_name = METRIC_LABEL[metric_name]
        ax.set(xlabel=x_name, ylabel=y_name)
        # plt.xticks(rotation=45)

        # ax.add_artist(legend)
        figname = f"prevalence_results__est_{estimator_group}_{plot_type}_{metric_name}"
        if xscale == "log":
            figname += "_xlog"

        dir2prevalence_results = dir2results / figname
        save_figure_to_folders(dir2prevalence_results)
# %%
# # What decay is better ?


##
# what decays are better ? always 1 day decay ! strange
metric = "average_precision_score"
sns.boxplot(
    all_results.loc[all_results["estimator"] == "random_forests"],
    x="decay_label",
    y=metric,
    hue="featurizer",
)
