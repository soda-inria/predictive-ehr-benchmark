from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from medem.constants import DIR2DOCS_EXPERIENCES, DIR2EXPERIENCES
from medem.experiences.configurations import FEATURIZER_COUNT_RANDOM_PROJ
from medem.experiences.utils import ICD10_LABEL2CHAPTER
from medem.reports.efficiency_plot import plot_efficiency
from medem.reports.utils import (
    COLORMAP_FEATURIZER,
    DEMOGRAPHIC_LABELS,
    FEATURIZER_LABEL,
    METRIC_LABEL,
    MODEL_LABELS,
    XLABELS,
    annotate_icd10,
    get_experience_results,
)
from medem.utils import save_figure_to_folders

expe_configs = [
    # (
    #     "icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_decay7",
    #     "estimator_label",
    # ),
    # (
    #     "icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_decay30",
    #     "estimator_label",
    # ),
    # (
    #     "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_3_models_decay7",
    #     "estimator_label",
    # ),
    #(
    #    "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_restricted_to_cui_voc",
    #    "estimator_label",
    #),
    (
        "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis\@cim10lvl_1__rs_0__min_prev_0.01_hospital_split"
        "estimator_label"
    )
]

sns.set(font_scale=1.3, style="whitegrid")  # , context="talk")

expe_name = "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_hospital_split"
dir2expe = Path(DIR2EXPERIENCES / expe_name)
expe_logs = get_experience_results(dir2expe)
expe_logs

expe_logs[["estimator", "featurizer","n_person_subtrain"]].value_counts()


@pytest.mark.parametrize("expe_name, col_order", expe_configs)
def test_report_expe(expe_name: str, col_order: str):
    dir2expe = Path(DIR2EXPERIENCES / expe_name)
    dir2results = DIR2DOCS_EXPERIENCES / expe_name
    dir2results.mkdir(exist_ok=True, parents=True)

    expe_logs = get_experience_results(dir2expe)
    test_prevalences = expe_logs.loc[
        :, expe_logs.columns.str.startswith("test_prevalence_")
    ].mean(axis=0)
    top_prevalences = test_prevalences[test_prevalences >= 10].index

    metric_name = "roc_auc_score"

    for chapter_name in top_prevalences:
        chapter_name_ = chapter_name.replace("test_prevalence_", "")
        chapter_id = ICD10_LABEL2CHAPTER[chapter_name_]
        g = plot_efficiency(
            expe_logs=expe_logs,
            metric_name="roc_auc_score",
            col_order="estimator_label",
            phenotype_chapter=chapter_id,
            estimators_list=[
                "ridge",
                "random_forests",
                "hist_gradient_boosting",
            ],
        )
        test_prevalence_ = test_prevalences[chapter_name]
        g.map_dataframe(
            annotate_icd10,
            **{"label": chapter_name_, "test_prevalence": test_prevalence_},
        )
        save_figure_to_folders(dir2results / f"{metric_name}__c_{chapter_id}")
