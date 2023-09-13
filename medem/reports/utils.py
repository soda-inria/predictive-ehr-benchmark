from pathlib import Path
from matplotlib import pyplot as plt
import re
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import pandas as pd
import seaborn as sns
import textwrap
from medem.constants import DIR2DATA, DIR2EXPERIENCES

from medem.experiences.configurations import (
    FEATURIZER_COUNT,
    FEATURIZER_COUNT_RANDOM_PROJ,
    FEATURIZER_COUNT_SVD,
    FEATURIZER_COUNT_WO_DECAY,
    FEATURIZER_CUI2VEC,
    FEATURIZER_CUI2VEC_SVD,
    FEATURIZER_DEMOGRAPHICS,
    FEATURIZER_EVENT2VEC_COMPLEMENTARY,
    FEATURIZER_EVENT2VEC_TRAIN,
    FEATURIZER_EVENT2VEC_TRAIN_WO_DECAY,
    FEATURIZER_SNDS,
    FEATURIZER_SNDS_SVD,
    CEHR_BERT_LABEL,
    FEATURIZER_SNDS_WO_DECAY,
    cohort_configuration_to_str,
)
from medem.experiences.utils import ICD10_LABEL2CHAPTER

TAB_COLOR = sns.color_palette("tab20", n_colors=20)

CEHR_BERT_LABEL_CLEAN = "Transformer-based (CEHR-BERT)"
FEATURIZER_COUNT_LABEL_CLEAN = "Decayed counting"

COLORMAP_FEATURIZER = {
    FEATURIZER_EVENT2VEC_TRAIN: TAB_COLOR[2],
    FEATURIZER_SNDS: TAB_COLOR[4],
    FEATURIZER_SNDS_SVD: TAB_COLOR[16],
    FEATURIZER_CUI2VEC: TAB_COLOR[8],
    FEATURIZER_CUI2VEC_SVD: TAB_COLOR[7],
    FEATURIZER_COUNT: TAB_COLOR[0],
    FEATURIZER_COUNT_LABEL_CLEAN: TAB_COLOR[0],
    FEATURIZER_COUNT_WO_DECAY: TAB_COLOR[18],
    FEATURIZER_COUNT_SVD: TAB_COLOR[18],
    CEHR_BERT_LABEL_CLEAN: TAB_COLOR[12],
    FEATURIZER_EVENT2VEC_TRAIN_WO_DECAY: TAB_COLOR[6],
    FEATURIZER_SNDS_WO_DECAY: TAB_COLOR[16],
    FEATURIZER_DEMOGRAPHICS: TAB_COLOR[10],
    # FEATURIZER_COUNT_RANDOM_PROJ: TAB_COLOR[6],
}
ESTIMATOR_STYLES = {
    "ridge": (1, 1),  # dotted
    "random_forests": (3, 2),  # dashed
    "hist_gradient_boosting": (3, 5, 1, 5),  # "dashdot",
    CEHR_BERT_LABEL: (1, 0),  # "solid"
}

METRIC_LABEL = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "roc_auc_score": "ROC AUC",
    "average_precision_score": "Average precision",
    "brier_score_loss": "Brier score",
}

XLABELS = {
    "subtrain_size": "Effective train ratio",
    "input_n_person_subtrain": "Train number of patients",
    "n_person_subtrain": "Number of patients in train set",
}
FEATURIZER_LABEL = "Featurizers"

MODEL_LABELS = {
    "ridge": "Logistic regression",
    "random_forests": "Random Forest",
    "hist_gradient_boosting": "Boosting",
    CEHR_BERT_LABEL: CEHR_BERT_LABEL_CLEAN,
}

DEMOGRAPHIC_LABELS = {
    0: "No demographics",
    3: "Age, sex and admission reason",
    6: "Age, sex, admission reason, destination, discharge type and value",
}
DECAY_LABELS = {
    "[0]": "No decay",
    "[0, 1]": "No decay and 1 day decay",
    "[0, 7]": "no decay and 7 days decay",
    "[0, 30]": "no decay and 30 days decay",
    "[0, 90]": "no decay and 90 days decay",
}


def get_experience_results(folder: str, long_format: bool = True):
    folder_ = Path(folder)
    if folder_.is_dir():
        run_logs = pd.concat(
            [pd.read_csv(f) for f in folder_.iterdir() if f.suffix == ".csv"]
        )
    else:
        raise ValueError(f"Folder {folder} does not exist")
    run_logs["estimator_label"] = run_logs["estimator"].apply(
        lambda x: MODEL_LABELS.get(x, x)
    )
    run_logs["featurizer"] = run_logs["featurizer"].apply(
        lambda x: FEATURIZER_COUNT_LABEL_CLEAN if x == FEATURIZER_COUNT else x
    )
    run_logs["demographic_label"] = run_logs["n_demographics"].apply(
        lambda x: DEMOGRAPHIC_LABELS.get(x, x)
    )
    if (
        "pipeline_best_params_event_transformer__decay_half_life_in_days"
        in run_logs.columns
    ):
        run_logs["decay_label"] = run_logs[
            "pipeline_best_params_event_transformer__decay_half_life_in_days"
        ].apply(lambda x: DECAY_LABELS.get(x, x))
    else:
        run_logs["decay_label"] = "No decay"
    if long_format:
        # pass to long format for chapter and score:
        scores_str = [
            "roc_auc_score",
            "average_precision_score",
            "accuracy_score",
            "brier_score_loss",
        ]
        id_vars = [
            "splitting_rs",
            "estimator",
            "featurizer",
            "n_demographics",
            "n_person_subtrain",
            "compute_time",
            "estimator_label",
            "demographic_label",
            "decay_label",
        ]
        embedding_results_l = []
        for score_ in scores_str:
            value_vars = [
                col
                for col in run_logs.columns
                if (col.startswith("c_") and col.endswith(score_))
            ]
            df_long = pd.melt(
                run_logs,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="target_label",
                value_name=score_,
            )
            df_long["target_label"] = df_long["target_label"].apply(
                lambda x: re.search("c_(\d\d*)__", x).group(1)
            )
            df_long = df_long.set_index([*id_vars, "target_label"])
            embedding_results_l.append(df_long)
        embedding_results_long = pd.concat(
            embedding_results_l, axis=1
        ).reset_index()
        return embedding_results_long
    else:
        return run_logs


def annotate_icd10(data, **kwargs):
    label = "\n".join(
        textwrap.wrap(kwargs["label"], 40, break_long_words=False)
    )
    test_prevalence_ = kwargs["test_prevalence"]
    ax = plt.gca()
    ax.text(
        0.95,
        0.05,
        label + f"\n Test prevalence: {test_prevalence_:.1f}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
    )


def get_cehr_bert_results_prognosis(
    cohort_config: str, model_name: str, test_prevalences: pd.DataFrame
):
    """
    the results from cehr_bert are stored in a different folder than experiment so are not
    git. This small function fix the problem by copying them into the
    experiences folder.
    """
    cohort_name = cohort_configuration_to_str(cohort_config)
    dir2cohort = DIR2DATA / cohort_name
    dir2expe_results = DIR2EXPERIENCES / (
        cohort_name + "_" + model_name + ".csv"
    )
    # if dir2expe_results.exists():
    #    target_metrics = pd.read_csv(dir2expe_results)
    # else:
    dir2evaluation = dir2cohort / "evaluation_train_val_split"

    target_prevalences = (
        pd.DataFrame.from_dict(
            ICD10_LABEL2CHAPTER, orient="index", columns=["target"]
        )
        .reset_index(names="icd10 chapter")
        .merge(test_prevalences)
    )
    target_prevalences.set_index("target").transpose()
    target_metrics_list = []
    # targets_computed = target_prevalences.target.astype(str)
    potential_evals_dir = list(dir2evaluation.iterdir())
    potential_evals_dir = [
        folder_
        for folder_ in potential_evals_dir
        if folder_.name.startswith(f"{model_name}_pr")
    ]
    for folder_ in potential_evals_dir:
        regex_ = re.search(
            f"{model_name}_pr(\d\d*)_pipeline__target_(\d\d*)", folder_.name
        )
        if regex_:
            pr_ = regex_.group(1)
            target_ = regex_.group(2)
            # for target_ in targets_computed:
            dir2metric = folder_ / "metrics"
            # if dir2metric.exists():
            metric_l = []
            # hack to deal with mixed float and int (in percentage)
            for f in dir2metric.iterdir():
                metric_l.append(pd.read_parquet(f))
            metrics_ = pd.concat(metric_l)
            metrics_["target"] = str(target_)
            target_metrics_list.append(metrics_)
    target_metrics = target_prevalences.merge(
        pd.concat(target_metrics_list), on="target", how="inner"
    )
    target_metrics.to_csv(dir2expe_results, index=False)
    target_metrics["estimator"] = CEHR_BERT_LABEL
    target_metrics["featurizer"] = MODEL_LABELS[CEHR_BERT_LABEL]
    target_metrics["estimator_label"] = MODEL_LABELS[CEHR_BERT_LABEL]
    if "brier_score_loss" not in target_metrics.columns:
        target_metrics["brier_score_loss"] = np.nan
    return target_metrics.rename(
        columns={
            "roc_auc_score": "roc_auc_score",
            "average_precision_score": "pr_auc",
        }
    )


def get_cehr_bert_results(cohort_config: str, model_name: str):
    cohort_name = cohort_configuration_to_str(cohort_config)
    dir2expe_results = DIR2EXPERIENCES / (
        cohort_name + "_" + model_name + ".csv"
    )
    if dir2expe_results.exists():
        cehr_bert_results = pd.read_csv(dir2expe_results)
    else:
        dir2cohort = DIR2DATA / cohort_name
        dir2evaluation = Path(dir2cohort / "evaluation_train_val_split")
        dir2metric = dir2evaluation / f"{model_name}" / "metrics"
        if dir2metric.exists():
            metric_l = []
            # hack to deal with mixed float and int (in percentage)
            for f in dir2metric.iterdir():
                metric_l.append(pd.read_parquet(f))

        cehr_bert_results = pd.concat(metric_l)

    cehr_bert_results["estimator"] = CEHR_BERT_LABEL
    cehr_bert_results["featurizer"] = CEHR_BERT_LABEL
    cehr_bert_results["estimator_label"] = CEHR_BERT_LABEL
    if "brier_score_loss" not in cehr_bert_results.columns:
        cehr_bert_results["brier_score_loss"] = np.nan
    return cehr_bert_results.rename(
        columns={
            "roc_auc": "roc_auc_score",
            "pr_auc": "average_precision_score",
        }
    )


def get_legend_handles_labels(featurizer_to_plots, estimator_to_plots):
    """From the lists of featurizers and estimators to plot, return the
    handles and labels for the legend.

    Args:
        featurizer_to_plots (_type_): _description_
        estimator_to_plots (_type_): _description_

    Returns:
        _type_: handles, labels
    """
    lw = 2
    plotted_featurizer_colors = {
        k: c_
        for k, c_ in COLORMAP_FEATURIZER.items()
        if k in featurizer_to_plots
    }
    empty_handle = [Line2D([0], [0], color="white")]
    handles_color = empty_handle + [
        Line2D([0], [0], color=c_, lw=2)
        for k, c_ in plotted_featurizer_colors.items()
    ]
    plotted_estimators_styles = {
        k: l_
        for k, l_ in ESTIMATOR_STYLES.items()
        if (k in estimator_to_plots) and (k != CEHR_BERT_LABEL)
    }
    estimator_color = "grey"
    lines = {
        k: Line2D([0], [0], color=estimator_color, lw=2, dashes=l_)
        for k, l_ in plotted_estimators_styles.items()
    }
    barline = LineCollection(np.empty((2, 2, 2)), color=estimator_color)
    ridge_handle = ErrorbarContainer(
        (lines["ridge"], [lines["ridge"]], [barline]), has_yerr=True
    )
    other_estimator_handle = [
        (Patch(alpha=0.3, color=estimator_color), line)
        for k, line in lines.items()
        if k != "ridge"
    ]
    handles_linestyle = empty_handle + [ridge_handle, *other_estimator_handle]
    labels_color = [FEATURIZER_LABEL] + list(plotted_featurizer_colors.keys())
    label_linestyle = [
        "Estimators",
        MODEL_LABELS["ridge"],
        *[MODEL_LABELS[k] for k, line in lines.items() if k != "ridge"],
    ]
    handles = handles_color + handles_linestyle
    labels = labels_color + label_linestyle
    return handles, labels
