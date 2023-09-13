import argparse
from copy import deepcopy
from typing import Dict
from datetime import datetime
from pathlib import Path
import numpy as np

import pandas as pd
from loguru import logger
from sklearn.model_selection import (
    GroupKFold,
    ParameterGrid,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
import os, psutil

from medem.constants import (
    COLNAME_INCLUSION_EVENT_START,
    COLNAME_OUTCOME,
    DIR2DATA,
    DIR2EXPERIENCES,
    COLNAME_PERSON,
)
from medem.experiences.configurations import (
    CONFIG_MACE_COHORT,
    CONFIG_MACE_ESTIMATION,
    FEATURIZER_EVENT2VEC_TRAIN,
    cohort_configuration_to_str,
)
from medem.experiences.features import get_date_details

from event2vec.event_transformer import (
    build_vocabulary,
    restrict_to_vocabulary,
    DemographicsTransformer,
)
from event2vec.svd_ppmi import event2vec
from medem.experiences.utils import (
    compute_person_subsample,
    config_experience2str,
    get_scores,
)
from medem.experiences.cohort import EventCohort
from medem.utils import add_age, to_lazyframe, to_pandas


def run_experience(config: Dict[str, str] = None):
    """
    This script runs experience for different cohort/tasks studying the effects
    of transferring the models between two groups of hospitals on the task
    performances. The performances of different (featurizers, models) are
    compared as well as the effective train size.

    Args:
        config (Dict[str, str], optional): _description_. Defaults to None.
    """
    config_cohort = CONFIG_MACE_COHORT
    config_experience = CONFIG_MACE_ESTIMATION
    logger.info(f"🧐 Running experience for {config_experience}\n-------------")

    cohort_name = cohort_configuration_to_str(config_cohort)
    # load data
    dir2cohort = DIR2DATA / cohort_name
    time_hash_xp = str(hash(datetime.now().strftime("%Y%m%d_%H%M%S")))
    if config.get("dir2experience", None) is None:
        dir2experience = DIR2EXPERIENCES / (
            "timesplit__" + cohort_name + "_hash_" + time_hash_xp
        )
    else:
        dir2experience = Path(config["dir2experience"])
    dir2experience.mkdir(exist_ok=True, parents=True)
    # TODO: remove if sufficient memory and replace by
    # start replacement
    # event_cohort = EventCohort(folder=dir2cohort)
    n_sample = 4000
    person = pd.read_parquet(
        dir2cohort / f"person_{n_sample}_non_cases.parquet"
    )
    event = pd.read_parquet(dir2cohort / f"event_{n_sample}_non_cases.parquet")
    event_cohort = EventCohort(person=person, event=event)
    # end replacement
    n_person_raw = event_cohort.person.shape[0]
    n_event_raw = event_cohort.event.shape[0]
    # fix vocabulary depending on full data
    logger.info(
        f"Original number of:\n - persons: {n_person_raw}\n - events {n_event_raw}"
    )
    if config_experience.get("path2vocabulary", None) is not None:
        study_vocabulary = list(
            pd.read_parquet(config_experience["path2vocabulary"]).columns.values
        )
    else:
        study_vocabulary = build_vocabulary(
            event=event_cohort.event,
            n_min_events=config_experience["n_min_events"],
        )
    restricted_event = restrict_to_vocabulary(
        event=event_cohort.event,
        vocabulary=study_vocabulary,
    )
    restricted_person = event_cohort.person.merge(
        restricted_event[COLNAME_PERSON].drop_duplicates(),
        on=COLNAME_PERSON,
        how="inner",
    )
    logger.info(
        f"Restricted number of:\n - persons: {len(restricted_person)}\n - events {len(restricted_event)}"
    )
    # adding static features
    static_features = get_date_details(
        restricted_person, colname_datetime=COLNAME_INCLUSION_EVENT_START
    ).drop("inclusion_time_of_day", axis=1)
    static_features = add_age(
        df=static_features,
        ref_datetime=COLNAME_INCLUSION_EVENT_START,
        colname_age="age_at_inclusion_event_start",
    )
    restricted_person_w_statics = static_features
    # prepare the runs configurations
    experience_grid_dict = {
        "estimator_config": config_experience["estimator_config"],
        "featurizer_config": config_experience["featurizer_config"],
        "subtrain_size": config_experience["subtrain_size"],
        "splitting_rs": config_experience["splitting_rs"],
        "colname_demographics": config_experience["colname_demographics"],
        "n_min_events": [config_experience["n_min_events"]],
    }
    # ( determinist) dataset splits (eg. temporal or by hospital)
    dataset_split = pd.read_parquet(dir2cohort / "dataset_split.parquet")
    train_person = restricted_person_w_statics.merge(
        dataset_split.loc[dataset_split["dataset"] == "train"],
        on="person_id",
        how="inner",
    )
    test_person = restricted_person_w_statics.merge(
        dataset_split.loc[dataset_split["dataset"] == "external_test"],
        on="person_id",
        how="inner",
    )
    # subsample tests to be able to evaluate with RF.
    test_person = test_person.sample(10000, random_state=42)
    nb_person_after_split = train_person.shape[0] + test_person.shape[0]

    expe_logs = {
        v: k
        for v, k in config_experience.items()
        if v not in experience_grid_dict.keys()
    }
    run_to_be_launch = list(ParameterGrid(experience_grid_dict))
    logger.info(f"Launch {len(run_to_be_launch)} run")
    for run_config in run_to_be_launch:
        logger.info(
            f"\n---------------------\n🚀 Begin run with parameters:\n {run_config}"
        )
        t0 = datetime.now()
        logger.info(f"Number of persons after split: {nb_person_after_split}")
        # validation and subtrain split
        if config_experience["validation_size"] is not None:
            validation_person = compute_person_subsample(
                person=train_person,
                train_size=config_experience["validation_size"],
                random_seed=run_config["splitting_rs"],
            )
            train_person_wo_validation = train_person.loc[
                ~train_person[COLNAME_PERSON].isin(
                    validation_person[COLNAME_PERSON]
                )
            ]
        else:
            train_person_wo_validation = train_person
        subtrain_person = compute_person_subsample(
            person=train_person_wo_validation,
            train_size=run_config["subtrain_size"],
            positive_ratio=None,  # remove the positive ratio constraint since I use a part of the population
            random_seed=run_config["splitting_rs"],
        )
        # Prepare static features information
        static_features_config = run_config.get("colname_demographics", None)
        if static_features_config is not None:
            static_features_categorical = static_features_config["categorical"]
            static_features_numerical = static_features_config["numerical"]
        else:
            static_features_categorical = []
            static_features_numerical = []

        colname_demographics = [
            *static_features_categorical,
            *static_features_numerical,
        ]
        # categorical_preprocessor = OneHotEncoder(handle_unknown="infrequent_if_exists", min_frequencies)
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler()
        column_transformer = ColumnTransformer(
            [
                (
                    "one-hot-encoder",
                    categorical_preprocessor,
                    static_features_categorical,
                ),
                (
                    "standard_scaler",
                    numerical_preprocessor,
                    static_features_numerical,
                ),
            ],
            remainder="passthrough",
        )
        # set the featurizer
        featurizer = run_config["featurizer_config"]["featurizer"]
        featurizer.set_params(**{"event": to_lazyframe(restricted_event)})
        featurizer.set_params(**{"colname_demographics": colname_demographics})
        if not isinstance(featurizer, DemographicsTransformer):
            featurizer.set_params(**{"vocabulary": study_vocabulary})
            featurizer.set_params(
                **{"n_min_events": run_config["n_min_events"]}
            )
        # Special case for local event2vec (to avoid recomputing the embeddings in the random search)
        if (
            run_config["featurizer_config"]["featurizer_name"]
            == FEATURIZER_EVENT2VEC_TRAIN
        ):
            local_embeddings = event2vec(
                events=to_pandas(
                    to_lazyframe(restricted_event).join(
                        to_lazyframe(subtrain_person).select(COLNAME_PERSON),
                        on=COLNAME_PERSON,
                        how="inner",
                    )
                ),
                # output_dir=DIR2DATA / "embeddings",
                **config_experience["local_embeddings_params"],
            )
            # event2vec(events=to_pandas(to_lazyframe(restricted_event).join(to_lazyframe(subtrain_person),on=COLNAME_PERSON,how="inner")),**config_experience["local_embeddings_params"])
            featurizer.set_params(**{"embeddings": local_embeddings})
        pipeline_steps = [
            ("event_transformer", featurizer),
            ("column_transformer", column_transformer),
        ]
        estimator = run_config["estimator_config"].estimator
        subtrain_prevalence = subtrain_person[COLNAME_OUTCOME].mean()
        test_prevalence = test_person[COLNAME_OUTCOME].mean()

        if config_experience.get("positive_ratio", None) is not None:
            estimator.set_params(
                **{
                    "class_weight": {
                        0: 1 - subtrain_prevalence,
                        1: subtrain_prevalence,
                    }
                }
            )
        estimator_kwargs = deepcopy(
            run_config["estimator_config"].estimator_kwargs
        )
        if estimator_kwargs is not None:
            if "featurizer_kwargs" in run_config["featurizer_config"].keys():
                estimator_kwargs.update(
                    run_config["featurizer_config"]["featurizer_kwargs"]
                )
        ## hospital splits
        # some are nan, replace by a random number
        na_groups = subtrain_person["first_care_site_id"].isna()
        subtrain_person.loc[na_groups, "first_care_site_id"] = "8312006712.0"
        train_groups = subtrain_person["first_care_site_id"].values
        gkf = GroupKFold(n_splits=5)
        splits = gkf.split(subtrain_person[COLNAME_PERSON], groups=train_groups)
        safe_splits = []
        for train_ix_, test_ix_ in splits:
            train_ix_[0] = 0
            safe_splits.append((train_ix_, test_ix_))

        pipeline_steps.append(("estimator", estimator))
        pipeline = RandomizedSearchCV(
            estimator=Pipeline(pipeline_steps),
            param_distributions=estimator_kwargs,
            n_iter=config_experience["randomsearch_n_iter"],
            scoring=config_experience["randomsearch_scoring"],
            random_state=config_experience["randomsearch_rs"],
            n_jobs=-1,
            pre_dispatch=2,
            cv=safe_splits,
        )

        # Fit estimator and predict
        logger.info(f"Fit pipeline: {pipeline}")
        pipeline.fit(
            X=subtrain_person[[COLNAME_PERSON, *colname_demographics]],
            y=subtrain_person[COLNAME_OUTCOME],
        )
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024**3)
        logger.info(f"Memory usage:{memory_usage} GB \n----------")

        # train scores
        train_y_prob = pipeline.predict_proba(
            subtrain_person[[COLNAME_PERSON, *colname_demographics]]
        )
        train_scores = get_scores(
            y_true=subtrain_person[COLNAME_OUTCOME],
            y_prob=train_y_prob,
        )
        train_scores = {"train_" + k: v for k, v in train_scores.items()}

        # test scores
        test_y_prob = pipeline.predict_proba(
            test_person[[COLNAME_PERSON, *colname_demographics]]
        )
        test_scores = get_scores(
            y_true=test_person[COLNAME_OUTCOME], y_prob=test_y_prob
        )

        # validation scores
        if config_experience["validation_size"] is not None:
            validation_y_prob = pipeline.predict_proba(
                validation_person[[COLNAME_PERSON, *colname_demographics]]
            )
            validation_scores = get_scores(
                y_true=validation_person[COLNAME_OUTCOME],
                y_prob=validation_y_prob,
            )
            validation_scores = {
                "validation_" + k: v for k, v in validation_scores.items()
            }
            validation_prevalence = {
                "validation_prevalence": 100
                * validation_person[COLNAME_OUTCOME].mean()
            }
            expe_logs = {
                **expe_logs,
                **validation_scores,
                **validation_prevalence,
                "n_persons_validation": validation_person.shape[0],
            }
        # logging
        t_log = datetime.now()
        logging_parameters_all = {
            **expe_logs,
            "estimator": run_config["estimator_config"].estimator_name,
            "featurizer": run_config["featurizer_config"].get(
                "featurizer_name", "naive"
            ),
            "subtrain_size": run_config["subtrain_size"],
            "splitting_rs": run_config["splitting_rs"],
            "colname_demographics": colname_demographics,
            "n_demographics": len(colname_demographics),
            "positive_ratio": config_experience.get("positive_ratio", ""),
            **test_scores,
            **train_scores,
            "n_person_raw": n_person_raw,
            "n_event_raw": n_event_raw,
            "n_person_test": test_person.shape[0],
            "compute_time": t_log - t0,
            "n_codes_raw": len(study_vocabulary),
            **{"test_prevalence": 100 * test_prevalence},
            **{"train_prevalence": 100 * subtrain_prevalence},
            "vocabulary": study_vocabulary,
        }
        #
        pipeline_best_params = pipeline.best_estimator_.get_params()
        pipeline_best_params_logged = {
            f"pipeline_best_params_{k}": pipeline_best_params.get(k, "")
            for k in [
                "estimator",
                "event_transformer__colname_demographics",
                "event_transformer__decay_half_life_in_days",
            ]
        }
        if not isinstance(featurizer, DemographicsTransformer):
            n_codes_featurizer = len(featurizer.vocabulary)
        else:
            n_codes_featurizer = 0

        logging_parameters_all.update(
            {
                "n_person_subtrain": subtrain_person.shape[0],
                "n_codes_featurizer": n_codes_featurizer,
                **pipeline_best_params_logged,
            }
        )
        # save results as one csv line
        run_name = (
            config_experience2str(logging_parameters_all)
            + "__"
            + str(hash(t0.strftime("%Y-%m-%d-%H-%M-%S")))
        )
        logging_pd = pd.DataFrame.from_dict(
            logging_parameters_all, orient="index"
        ).transpose()
        logging_pd.to_csv(dir2experience / f"{run_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--xp_name", type=str,default=None,help="xp folder to consolidate",)
    parser.add_argument(
        "--dir2experience",
        type=str,
        default=None,
        help="Path to train results, default is package data directory.",
    )
    config, _ = parser.parse_known_args()
    config = vars(config)
    logger.remove(0)
    logger.add("mace_out.log")
    run_experience(config=config)
