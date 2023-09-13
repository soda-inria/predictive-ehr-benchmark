from datetime import datetime
from typing import Dict, List, Tuple, Union
import numpy as np

import pandas as pd
from joblib import Memory
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_INCLUSION_DAY_OF_WEEK,
    COLNAME_INCLUSION_MONTH,
    COLNAME_INCLUSION_TIME_OF_DAY,
    COLNAME_OUTCOME,
    COLNAME_PERSON,
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    COLNAME_START,
    COLNAME_TARGET_CODE,
    DIR2CACHE,
)

memory = Memory(DIR2CACHE)


def config_experience2str(config: Dict):
    return f"tr_{100*config['subtrain_size']:2.0f}__feat_{config['featurizer']}__estimator_{config['estimator']}__searchrs_{config['randomsearch_rs']}"


def compute_person_subsample(
    person: pd.DataFrame,
    train_size: float,
    positive_ratio: float = None,
    random_seed: int = None,
) -> pd.DataFrame:
    """Subsample a person dataframe to obtain a more balanced training set with respect to class labels.
    The returned dataframe is a subset of the person dataframe of size `len(person)*train_size`
    and with positive class ratio `positive_ratio`.

    Works only for binary classification.

    Args:
        person (pd.DataFrame): _description_
        train_size (float): _description_
        subsample_positive_ratio (float, optional): Target positive ratio. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    N_train = len(person)
    if person[COLNAME_OUTCOME].nunique() != 2:
        raise ValueError(f"Outcome should be binary.")
    mask_positive = person[COLNAME_OUTCOME] == 1
    if positive_ratio is None:
        positive_ratio = mask_positive.sum() / N_train
    n_subtrain_positive = np.min(
        [
            int(train_size * positive_ratio * N_train),
            mask_positive.sum(),
        ],
    )
    n_subtrain_negative = int(train_size * (1 - positive_ratio) * N_train)

    sub_train_positive = person[mask_positive].sample(
        n_subtrain_positive, random_state=random_seed
    )
    sub_train_negative = person[~mask_positive].sample(
        n_subtrain_negative, random_state=random_seed
    )
    subsample = (
        pd.concat([sub_train_negative, sub_train_positive])
        .sample(frac=1, random_state=random_seed)
        .reset_index()
    )
    return subsample


# TODO: might be part of the cohort object
def event_train_test_split(
    event_cohort,
    train_size: float = None,
    test_size: float = None,
    random_seed: int = None,
):
    """Return train_event, test_event, train_person, test_person

    Args:
        event_cohort (EventCohort): _description_
        train_size (float, optional): _description_. Defaults to None.
        test_size (float, optional): _description_. Defaults to None.
        random_seed (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    person_ids = event_cohort.person[COLNAME_PERSON]
    if train_size == 1:
        train_person_id = person_ids.sample(frac=1.0, random_state=random_seed)
        test_person = pd.DataFrame()
        test_event = pd.DataFrame()
    else:
        train_person_id, test_person_id = train_test_split(
            person_ids,
            train_size=train_size,
            test_size=test_size,
            random_state=random_seed,
        )
        test_person = event_cohort.person.merge(
            pd.DataFrame(test_person_id, columns=[COLNAME_PERSON]),
            on=COLNAME_PERSON,
            how="inner",
        )
        test_event = event_cohort.event.merge(
            pd.DataFrame(test_person_id, columns=[COLNAME_PERSON]),
            on=COLNAME_PERSON,
            how="inner",
        )
    train_event = event_cohort.event.merge(
        pd.DataFrame(train_person_id, columns=[COLNAME_PERSON]),
        on=COLNAME_PERSON,
        how="inner",
    )
    train_person = event_cohort.person.merge(
        pd.DataFrame(train_person_id, columns=[COLNAME_PERSON]),
        on=COLNAME_PERSON,
        how="inner",
    )

    return (
        train_event,
        test_event,
        train_person,
        test_person,
    )


# TODO: read evaluating ML models and their diagnostic value, Varoquaux et
# Colliot, 2023: https://hal.science/hal-03682454/document
# TODO: add recall@k ie. number of TP in top k predictions
def get_binary_scores(y_true: np.array, y_prob: np.array):
    """
    Args:
        y_true (_type_): size n_samples
        y_prob (_type_): Class 1 scores, of size n_samples

    Returns:
        _type_: _description_
    """
    y_prob_ = y_prob[:, 1]
    scores = {}
    scores[brier_score_loss.__name__] = brier_score_loss(
        y_true=y_true, y_prob=y_prob_
    )
    scores[roc_auc_score.__name__] = roc_auc_score(
        y_true=y_true, y_score=y_prob_
    )
    y_pred = (y_prob_ > 0.5).astype(int)
    scores[accuracy_score.__name__] = accuracy_score(
        y_true=y_true, y_pred=y_pred
    )
    scores[average_precision_score.__name__] = average_precision_score(
        y_true=y_true, y_score=y_prob_
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    scores["tn"] = tn
    scores["fp"] = fp
    scores["fn"] = fn
    scores["tp"] = tp
    return scores


def get_scores(y_true, y_prob, classes=None) -> Dict[str, float]:
    """Wrapper for both binary and multilabel tasks.

    Args:
        y_true (_type_): _description_
        y_prob (_type_): _description_
        classes (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if classes is None:
        return get_binary_scores(y_true, y_prob)
    else:
        scores = {}
        for i, c_ in enumerate(classes):
            class_scores = get_binary_scores(y_true[:, i], y_prob[i])
            for k, v in class_scores.items():
                scores[f"c_{c_}__score_{k}"] = v
        return scores


# ICD10 chapters
ICD10_CHAPTERS2LABEL = {
    "1": "Certain infectious and parasitic diseases",
    "2": "Neoplasms",
    "3": "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    "4": "Endocrine, nutritional and metabolic diseases",
    "5": "Mental, Behavioral and Neurodevelopmental disorders",
    "6": "Diseases of the nervous system",
    "7": "Diseases of the eye and adnexa",
    "8": "Diseases of the ear and mastoid process",
    "9": "Diseases of the circulatory system",
    "10": "Diseases of the respiratory system",
    "11": "Diseases of the digestive system",
    "12": "Diseases of the skin and subcutaneous tissue",
    "13": "Diseases of the musculoskeletal system and connective tissue",
    "14": "Diseases of the genitourinary system",
    "15": "Pregnancy, childbirth and the puerperium",
    "16": "Certain conditions originating in the perinatal period",
    "17": "Congenital malformations, deformations and chromosomal abnormalities",
    "18": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    "19": "Injury, poisoning and certain other consequences of external causes",
    "20": "External causes of morbidity",
    "21": "Factors influencing health status and contact with health services",
    "22": "Codes for special purposes",
}

ICD10_LABEL2CHAPTER = {v: k for k, v in ICD10_CHAPTERS2LABEL.items()}


def get_prognosis_prevalence(
    y: Union[pd.Series, np.array],
    cim10_nb_digits: int = 1,
    classes: List = None,
):
    if classes is None:
        # need to fit the MultiLabelBinarizer to get the classes
        target_vocabulary = y.explode().unique()
        mlb = MultiLabelBinarizer(classes=target_vocabulary)
        y_ = mlb.fit_transform(y)
        classes = mlb.classes_
    else:
        # uses existing classes
        y_ = y
    if cim10_nb_digits == 1:
        phenotypes_columns = [ICD10_CHAPTERS2LABEL[c] for c in classes]
        phenotype_df = pd.DataFrame(y_, columns=phenotypes_columns)
        phenotype_prevalences = (
            (100 * phenotype_df.sum(axis=0) / len(phenotype_df))
            .sort_values(ascending=False)
            .to_frame("prevalence")
            .transpose()
        )
    else:
        raise NotImplementedError
    return phenotype_prevalences


MORNING = "morning"
AFTERNOON = "afternoon"
NIGHT = "night"


def get_time_of_day(timestamp: datetime) -> str:
    if (timestamp.hour < 12) and (timestamp.hour >= 7):
        return MORNING
    elif (timestamp.hour >= 12) and (timestamp.hour < 20):
        return AFTERNOON
    else:
        return NIGHT


def get_date_details(
    person: pd.DataFrame, colname_datetime: str
) -> pd.DataFrame:
    person[COLNAME_INCLUSION_MONTH] = person[colname_datetime].dt.month
    person[COLNAME_INCLUSION_DAY_OF_WEEK] = person[
        colname_datetime
    ].dt.dayofweek
    person[COLNAME_INCLUSION_TIME_OF_DAY] = person[colname_datetime].apply(
        get_time_of_day
    )

    return person
