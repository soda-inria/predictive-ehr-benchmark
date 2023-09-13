from dataclasses import dataclass
from typing import Dict, List

from dateutil.parser import parse
import numpy as np
from sklearn.base import BaseEstimator

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from medem.constants import (
    COLNAME_SOURCE_CODE,
    COLNAME_START,
    DIR2DATA,
    DIR2EMBEDDINGS,
    DIR2RESOURCES,
    TASK_LOS_CATEGORICAL,
    TASK_MACE,
    TASK_PROGNOSIS,
    TASK_MORTALITY,
    TASK_REHOSPITALIZATION,
)

import event2vec.event_transformer as et

EVENT_CONFIG = {
    "procedure_occurrence": {
        COLNAME_START: "procedure_datetime",
        COLNAME_SOURCE_CODE: "procedure_source_value",
    },
    "drug_exposure_administration": {
        COLNAME_START: "drug_exposure_start_datetime",
        COLNAME_SOURCE_CODE: "drug_class_source_value",
    },  # corresponds to ATC code
    "condition_occurrence": {
        COLNAME_START: "condition_start_datetime",
        COLNAME_SOURCE_CODE: "condition_source_value",
    },
    "measurement": {
        COLNAME_START: "measurement_datetime",
        COLNAME_SOURCE_CODE: "measurement_source_value",
        "path2mapping": DIR2RESOURCES / "mapping_loinc2nabm_2020.csv",
    },
    "drug_exposure": {
        COLNAME_START: "start",
        COLNAME_SOURCE_CODE: "drug_class_source_value",
    },
}
# NOTE: the mapping should have only two columns:
# - `event_source_concept_id`, the  source code (e.g. LOINC code)
# -  `target_concept_id` the target code (e.g. NABM code)
#
#  Not using all possible event tables for now
DEFAULT_EVENT_TABLES = [
    "procedure_occurrence",
    "drug_exposure_administration",
    "condition_occurrence",
    # "measurement",
    # "drug_exposure_prescription",
]

I2B2_EVENT_TABLES = [
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
]

# For events, keep only the most frequent ones, min_frequency=0.05 used in OneHotEncoder
STATIC_FEATURES_DICT = {
    "categorical": [
        # "discharge_type_source_value",
        # "destination_source_value",
        "admitted_from_source_value",
        "discharge_to_source_value",
        "gender_source_value",
        "inclusion_event_source_concept_id",
    ],
    "numerical": ["age_at_inclusion_event_start"],
}
# Not considered
# "admission_type_source_value", same info than admission_reason_source_value but too aggregated
# "admitted_from_source_value", same info than admission_reason_source_value but less aggregated with long tail distrib.
# "provenance_source_value", almost same info than admission_reason_source_value but with the MCO/SSR distinction but with very little occurrences


@dataclass
class EstimatorConfig:
    estimator_name: str
    estimator: List[BaseEstimator]
    estimator_kwargs: Dict[str, List]

    def dict(self):
        # dic_ = {k: self.__getattribute__(k) for k in self.__dataclass_fields__}
        return {
            "estimator_name": self.estimator_name,
            "estimator": self.estimator,
            "estimator_kwargs": self.estimator_kwargs,
        }


def make_event_config(event_table_names: List[str]) -> Dict[str, Dict]:
    event_config_ = {k: EVENT_CONFIG[k] for k in event_table_names}
    return event_config_


DEFAULT_EVENT_CONFIG = make_event_config(DEFAULT_EVENT_TABLES)
# T1: LOS
CONFIG_LOS_COHORT = {
    "database_name": "cse210038_20220921_160214312112",
    "cohort_name": "complete_hospitalization_los",
    "study_start": parse("2017-01-01"),
    "study_end": parse("2022-06-01"),
    "min_age_at_admission": 18,
    "sup_quantile_visits": 0.95,
    "task_name": TASK_LOS_CATEGORICAL,
    "los_categories": np.array([0, 7, np.inf]),
    "horizon_in_days": 30,  # used for prediction tasks AND for the loss of follow-up compared to the study-end.
    #"deceased": "include",  # unsued for this task, to deprecate ?
    "event_tables": DEFAULT_EVENT_CONFIG,
    "test_size": 0.2,
    "n_min_events": 10,
}
# T2 : PROGNOSIS
CONFIG_PROGNOSIS_COHORT = {
    "database_name": "cse210038_20220921_160214312112",
    "cohort_name": "icd10_prognosis",
    # population_selection
    "study_start": parse("2017-01-01"),
    "study_end": parse("2022-06-01"),
    "min_age_at_admission": 18,
    "sup_quantile_visits": 0.95,
    "horizon_in_days": 7,  # used for prediction tasks AND for the loss of follow-up compared to the study-end.
    "n_min_visits": 2,
    "with_incomplete_hospitalization": True,
    "visits_w_billing_codes_only": True,
    # task
    "task_name": TASK_PROGNOSIS,
    "cim10_nb_digits": 1,
    "min_prevalence": 0.01,
    "visit_random_state": 0,
    "event_tables": DEFAULT_EVENT_CONFIG,
    "test_size": 0.2,
    "n_min_events": 10,
}
# T3: MACE
# To run on the omop 200K cohort, modify:
# database_name="cse210038_20220921_160214312112",
# database_type="omop"
# event_tables=DEFAULT_EVENT_CONFIG
CONFIG_MACE_COHORT = {
    "database_name": DIR2DATA / "mace_cohort",
    "database_type": "i2b2",
    "cohort_name": "mace",
    # population_selection
    "study_start": parse("2018-01-01"),
    "study_end": parse("2020-12-31"),
    "min_age_at_admission": 18,
    "sup_quantile_visits": 0.95,
    "horizon_in_days": 360,  # used for prediction tasks AND for the loss of follow-up compared to the study-end.
    "n_min_visits": 2,
    "with_incomplete_hospitalization": True,
    "visits_w_billing_codes_only": True,
    # task
    "task_name": TASK_MACE,
    "event_tables": make_event_config(I2B2_EVENT_TABLES),
    "lazy": True,
    "index_visit": "random",
    "n_min_events": 10,
    # split size
    "test_size": 0.2,
}


PATH2SNDS_EMBEDDINGS = (
    DIR2EMBEDDINGS
    / "snds"
    / "echantillon_mid_grain_r=90-centered2019-12-05_19:11:27.parquet"
)

PATH2CUI2VEC_FIRST_EMBEDDINGS = (
    DIR2EMBEDDINGS / "cui2vec" / "cui2vec_first.parquet"
)
PATH2CUI2VEC_MEAN_EMBEDDINGS = (
    DIR2EMBEDDINGS / "cui2vec" / "cui2vec_mean.parquet"
)


def cohort_configuration_to_str(config: Dict) -> str:
    task_name = config["task_name"]
    if task_name == TASK_LOS_CATEGORICAL:
        task_name += f"@{len(config['los_categories'])}"
    elif task_name in [TASK_REHOSPITALIZATION, TASK_MORTALITY, TASK_MACE]:
        task_name += f"@{config['horizon_in_days']}"
    elif task_name == TASK_PROGNOSIS:
        task_name += f"@cim10lvl_{config['cim10_nb_digits']}__rs_{config['visit_random_state']}__min_prev_{config['min_prevalence']}"
        target_chapter = config.get("target_chapter", None)
        if target_chapter is not None:
            task_name += f"chap_{target_chapter}"
    if config.get("index_visit", None) is not None:
        task_name += f"__index_visit_{config['index_visit']}"
    return f"{config['cohort_name']}__age_min_{config['min_age_at_admission']}__dates_{config['study_start'].year}_{config['study_end'].year}__task__{task_name}"


FEATURIZER_DEMOGRAPHICS = "Demographics"
FEATURIZER_COUNT = "Count Encoding"
FEATURIZER_COUNT_WO_DECAY = "Count Encoding wo decay"
FEATURIZER_COUNT_SVD = "Count Encoding + SVD"
FEATURIZER_COUNT_RANDOM_PROJ = "Count Encoding + random projection"
FEATURIZER_EVENT2VEC_TRAIN = "Local Embeddings"
FEATURIZER_EVENT2VEC_TRAIN_WO_DECAY = "Train Embeddings wo decay"
FEATURIZER_SNDS = "SNDS Embeddings"
FEATURIZER_SNDS_WO_DECAY = "SNDS Embeddings wo decay"
FEATURIZER_SNDS_SVD = "SNDS Embeddings + SVD"
FEATURIZER_CUI2VEC = "Cui2vec Embeddings"
FEATURIZER_CUI2VEC_SVD = "Cui2vec Embeddings + SVD"
FEATURIZER_EVENT2VEC_COMPLEMENTARY = "Complementary Embeddings"
CEHR_BERT_LABEL = "CEHR-BERT"

N_COMPONENTS = 30  # number of components for SVD or in-domain embeddings
N_MIN_EVENTS = 10  # minimum number of events for a code to be considered


GRID_DECAYS = [
    [0],
    [0, 1],
    # [0, 7],
    # [0, 30],
    # [0, 90],
]
# LOS ESTIMATION ##############################################################
FEATURIZERS_TASK_LOS = [
    {
        "featurizer_name": FEATURIZER_EVENT2VEC_TRAIN,
        "featurizer": et.Event2vecPretrained(event=None, embeddings=None),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    {
        "featurizer_name": FEATURIZER_DEMOGRAPHICS,
        "featurizer": et.DemographicsTransformer(
            event=None,
        ),
    },
    {
        "featurizer_name": FEATURIZER_SNDS,
        "featurizer": et.Event2vecPretrained(
            event=None,
            embeddings=PATH2SNDS_EMBEDDINGS,
        ),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    {
        "featurizer_name": FEATURIZER_COUNT,
        "featurizer": et.OneHotEvent(
            event=None,
        ),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
]
ESTIMATORS_TASK_LOS = [
    EstimatorConfig(
        estimator_name="ridge",
        estimator=LogisticRegression(class_weight="balanced", n_jobs=-1),
        estimator_kwargs={"estimator__C": [1e-3, 1e-2, 1e-1, 5e-1, 1]},
    ),
    EstimatorConfig(
        estimator_name="random_forests",
        estimator=RandomForestClassifier(n_jobs=-1),
        estimator_kwargs={
            "estimator__min_samples_leaf": [
                2,
                10,
                50,
                100,
                200,
            ],
            "estimator__n_estimators": [50, 100, 200, 500],
        },
    ),
    # EstimatorConfig(
    #     estimator_name="hist_gradient_boosting",
    #     estimator=HistGradientBoostingClassifier(early_stopping=True),
    #     estimator_kwargs={
    #         "estimator__learning_rate": [
    #             1e-3,
    #             1e-2,
    #             1e-1,
    #             1,
    #         ],
    #         "estimator__max_leaf_nodes": [10, 20, 30, 50],
    #     },
    # ),
]
LOS_SUBTRAIN_GRID = list([0.01, *list(np.arange(0.1, 1.1, step=0.2)), 1])
CONFIG_LOS_ESTIMATION = {
    "validation_size": None,  # no validation for los
    "subtrain_size": LOS_SUBTRAIN_GRID,
    "splitting_rs": list(range(5)),  # TODO: change to 5
    "estimator_config": ESTIMATORS_TASK_LOS,
    "featurizer_config": FEATURIZERS_TASK_LOS,
    "randomsearch_scoring": "roc_auc",
    "randomsearch_n_iter": 10,
    "randomsearch_rs": 0,
    "n_min_events": 10,
    "colname_demographics": [
        STATIC_FEATURES_DICT,
    ],
    "local_embeddings_params": {
        "colname_concept": COLNAME_SOURCE_CODE,
        "window_radius_in_days": 30,
        "window_orientation": "center",
        "backend": "pandas",
        "d": N_COMPONENTS,
    },
    # "path2vocabulary": PATH2CUI2VEC_FIRST_EMBEDDINGS,
}

# PROGNOSIS CONFIG ############################################################
FEATURIZERS_TASK_PROGNOSIS = [
    {
        "featurizer_name": FEATURIZER_EVENT2VEC_TRAIN,
        "featurizer": et.Event2vecPretrained(event=None, embeddings=None),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    {
        "featurizer_name": FEATURIZER_SNDS,
        "featurizer": et.Event2vecPretrained(
            event=None,
            embeddings=PATH2SNDS_EMBEDDINGS,
        ),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    {
        "featurizer_name": FEATURIZER_COUNT,
        "featurizer": et.OneHotEvent(
            event=None,
        ),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    {
        "featurizer_name": FEATURIZER_DEMOGRAPHICS,
        "featurizer": et.DemographicsTransformer(
            event=None,
        ),
    },
]

ESTIMATORS_PROGNOSIS = [
    # EstimatorConfig(
    #     estimator_name="ridge",
    #     estimator=LogisticRegression(class_weight="balanced", n_jobs=-1),
    #     estimator_kwargs={"estimator__C": [1e-3, 1e-2, 1e-1, 5e-1, 1]},
    # ),
    # EstimatorConfig(
    #     estimator_name="random_forests",
    #     estimator=RandomForestClassifier(n_jobs=-1),
    #     estimator_kwargs={
    #         "estimator__min_samples_leaf": [
    #             2,
    #             10,
    #             50,
    #             100,
    #             200,
    #         ],
    #         "estimator__n_estimators": [50, 100, 200, 500],
    #     },
    # ),
    EstimatorConfig(
        estimator_name="hist_gradient_boosting",
        estimator=HistGradientBoostingClassifier(early_stopping=True),
        estimator_kwargs={
            "estimator__learning_rate": [
                1e-3,
                1e-2,
                1e-1,
                1,
            ],
            "estimator__max_leaf_nodes": [10, 20, 30, 50],
        },
    ),
]
PROGNOSIS_SUBTRAIN_GRID = [0.5, 0.9]
CONFIG_PROGNOSIS_ESTIMATION = {
    "validation_size": None,  # No validation for prognosis
    "subtrain_size": PROGNOSIS_SUBTRAIN_GRID,
    "splitting_rs": list(range(3)),
    "estimator_config": ESTIMATORS_PROGNOSIS,
    "featurizer_config": FEATURIZERS_TASK_PROGNOSIS,
    "randomsearch_scoring": "roc_auc",
    "randomsearch_n_iter": 10,
    "randomsearch_rs": 0,
    "n_min_events": 10,
    "colname_demographics": [
        STATIC_FEATURES_DICT,
    ],
    "local_embeddings_params": {
        "colname_concept": COLNAME_SOURCE_CODE,
        "window_radius_in_days": 60,
        "window_orientation": "center",
        "backend": "pandas",
        "d": N_COMPONENTS,
    },
}

# MACE CONFIG #################################################################
GRID_DECAYS_MACE = [[0], [0, 1], [0, 7], [0, 30]]
FEATURIZERS_MACE = [
    # {
    #     "featurizer_name": FEATURIZER_DEMOGRAPHICS,
    #     "featurizer": et.DemographicsTransformer(
    #         event=None,
    #     ),
    # },
    # {
    #     "featurizer_name": FEATURIZER_EVENT2VEC_TRAIN,
    #     "featurizer": et.Event2vecPretrained(event=None, embeddings=None),
    #     "featurizer_kwargs": {
    #         "event_transformer__decay_half_life_in_days": GRID_DECAYS
    #     },
    # },
    {
        "featurizer_name": FEATURIZER_SNDS,
        "featurizer": et.Event2vecPretrained(
            event=None,
            embeddings=PATH2SNDS_EMBEDDINGS,
        ),
        "featurizer_kwargs": {
            "event_transformer__decay_half_life_in_days": GRID_DECAYS
        },
    },
    # {
    #     "featurizer_name": FEATURIZER_COUNT,
    #     "featurizer": et.OneHotEvent(
    #         event=None,
    #     ),
    #     "featurizer_kwargs": {
    #         "event_transformer__decay_half_life_in_days": GRID_DECAYS
    #     },
    # },
]
ESTIMATORS_MACE = [
    EstimatorConfig(
        estimator_name="ridge",
        estimator=LogisticRegression(n_jobs=-1, class_weight={0: 1, 1: 20}),
        estimator_kwargs={"estimator__C": [1e-3, 1e-2, 1e-1, 5e-1, 1]},
    ),
    # EstimatorConfig(
    #     estimator_name="random_forests",
    #     estimator=RandomForestClassifier(n_jobs=-1),
    #     estimator_kwargs={
    #         "estimator__min_samples_leaf": [
    #             2,
    #             10,
    #             50,
    #             100,
    #             200,
    #         ],
    #         "estimator__n_estimators": [50, 100, 200, 500],
    #     },
    # ),
    EstimatorConfig(
        estimator_name="hist_gradient_boosting",
        estimator=HistGradientBoostingClassifier(
            early_stopping=True, class_weight={0: 1, 1: 20}
        ),
        estimator_kwargs={
            "estimator__learning_rate": [
                1e-3,
                1e-2,
                1e-1,
                1,
            ],
            "estimator__max_leaf_nodes": [10, 20, 30, 50],
        },
    ),
]
MACE_SUBTRAIN_GRID = [0.99]  # [0.02, 0.1, 0.5, 0.9, 1]
CONFIG_MACE_ESTIMATION = {
    "validation_size": None,
    "subtrain_size": MACE_SUBTRAIN_GRID,
    "positive_ratio": None,  # setting the subsample positive ratio, set to None to deactivate
    "splitting_rs": [0, 1],
    "estimator_config": ESTIMATORS_MACE,
    "featurizer_config": FEATURIZERS_MACE,
    "randomsearch_scoring": "average_precision",  # more clinically relevant to put 'neg_brier_score' / or 'average_precision'
    "randomsearch_n_iter": 15,
    "randomsearch_rs": 0,
    "n_min_events": 10,
    "colname_demographics": [
        STATIC_FEATURES_DICT,
    ],
    "local_embeddings_params": {
        "colname_concept": COLNAME_SOURCE_CODE,
        "window_radius_in_days": 60,
        "window_orientation": "center",
        "backend": "pandas",
        "d": N_COMPONENTS,
    },
}

# CONFIG FOR scripts:
COHORT_NAME2CONFIG = {
    "los": CONFIG_LOS_COHORT,
    "prognosis": CONFIG_PROGNOSIS_COHORT,
    "mace": CONFIG_MACE_COHORT,
}

# ABLATION STUDIES################################################################
ESTIMATORS_ABLATION_LOS = [
    EstimatorConfig(
        estimator_name="hist_gradient_boosting",
        estimator=[HistGradientBoostingClassifier(early_stopping=True)],
        estimator_kwargs={
            "estimator__learning_rate": [
                1e-3,
                1e-2,
                1e-1,
                1,
            ],
            "estimator__max_leaf_nodes": [10, 20, 30, 50],
        },
    )
]
FEATURIZER_ABLATION_LOS = [
    {
        "featurizer_name": FEATURIZER_EVENT2VEC_TRAIN,
        "featurizer": et.Event2vecFeaturizer(
            event=None,
            output_dir=DIR2DATA / "embeddings",
            colname_code=COLNAME_SOURCE_CODE,
            window_radius_in_days=60,
            window_orientation="center",
            backend="pandas",
            d=N_COMPONENTS,
        ),
    },
    {
        "featurizer_name": FEATURIZER_SNDS,
        "featurizer": et.Event2vecPretrained(
            event=None,
            embeddings=PATH2SNDS_EMBEDDINGS,
            n_min_events=N_MIN_EVENTS,
        ),
    },
    {
        "featurizer_name": FEATURIZER_COUNT,
        "featurizer": et.OneHotEvent(
            event=None,
            n_min_events=N_MIN_EVENTS,
        ),
    },
]
ABLATION_SUBTRAIN_GRID = [0.1, 0.5, 1]
ABLATION_STATIC_FEATURES_DICT = {
    "categorical": [
        "admission_reason_source_value",
        "gender_source_value",
    ],
    "numerical": ["age_to_inclusion_event_start"],
}
# three decays is blowing up memory...
ABLATION_DECAYS = [[0], [0, 1], [0, 7], [0, 30], [0, 90]]
CONFIG_ABLATION_LOS = {
    "test_size": 0.3,
    "subtrain_size": ABLATION_SUBTRAIN_GRID,
    "splitting_rs": list(range(5)),
    "estimator_config": ESTIMATORS_ABLATION_LOS,
    "featurizer_config": FEATURIZER_ABLATION_LOS,
    "randomsearch_scoring": "roc_auc",
    "randomsearch_n_iter": 5,
    "randomsearch_rs": 0,
    "n_min_events": 10,
    "colname_demographics": [
        None,
        ABLATION_STATIC_FEATURES_DICT,
        STATIC_FEATURES_DICT,
    ],
    "decay_half_life_in_days": [[0, 7]],
}


# Transfer configurations
TEST_HOSPITALS_DICT = {
    "HOPITAL AVICENNE": 8312039899,
    "HOPITAL JEAN VERDIER": 8312041542,
    "HOPITAL AMBROISE PARE": 8312002245,
    "GH COCHIN": 8312052888,
    "GH RAYMOND POINCARE-BERCK": 8312016826,
}
