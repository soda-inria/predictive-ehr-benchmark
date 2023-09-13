from typing import List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer

from medem.constants import (
    COLNAME_FOLLOWUP_START,
    COLNAME_OUTCOME,
    COLNAME_PERSON,
    COLNAME_SOURCE_CODE,
    COLNAME_START,
)
from medem.preprocessing.utils import coarsen_cim10_to_chapter


def get_feature_sparsity(X: np.array):
    return (X == 0).sum() / (X.shape[0] * X.shape[1])


class NaivePrognosisBaseline(BaseEstimator):
    """Naive classifier predicting the same chapters as the one present in last
    visit.

    Args:
        EventTransformerMixin (_type_): _description_ BaseEstimator (_type_):
        _description_
    """

    def __init__(self, event: pd.DataFrame):
        self.event = event
        return

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self

    def predict(self, X):
        X_event = pd.DataFrame({COLNAME_PERSON: X}).merge(
            self.event, on=COLNAME_PERSON, how="inner"
        )
        mask_conditions = (
            X_event["event_source_type_concept_id"] == "condition_occurrence"
        )
        conditions = X_event.loc[mask_conditions]
        conditions_before_followup = conditions.loc[
            conditions["start"] <= conditions[COLNAME_FOLLOWUP_START]
        ]

        n_conditions_at_followup = (
            conditions_before_followup[COLNAME_START]
            == conditions_before_followup[COLNAME_FOLLOWUP_START]
        ).sum()
        if n_conditions_at_followup > 0:
            raise ValueError(
                f"Conditions in event features should precede followup: got {n_conditions_at_followup} occurring at followup timestamp."
            )
        # coarsen to chapter:
        conditions_before_followup = coarsen_cim10_to_chapter(
            diagnoses_df=conditions_before_followup.rename(
                columns={"event_source_concept_id": "condition_source_value"}
            )
        ).dropna(subset=["condition_source_value_coarse"])
        conditions_before_followup[
            "condition_source_value_coarse"
        ] = conditions_before_followup["condition_source_value_coarse"].astype(
            str
        )
        last_conditions_before_followup = (
            conditions_before_followup.sort_values(
                [COLNAME_PERSON, COLNAME_START]
            )
            .groupby(COLNAME_PERSON)["condition_source_value_coarse"]
            .agg(lambda x: np.unique(list(x)))
            .reset_index(name="y_pred")
        )
        naive_predictions = pd.DataFrame({COLNAME_PERSON: X}).merge(
            last_conditions_before_followup, on=COLNAME_PERSON, how="left"
        )
        # For people without any diagnoses codes at previous visit (strange but
        # could happen), predict a random chapter.
        mask_no_previous_condition = naive_predictions["y_pred"].isna()
        naive_predictions.loc[mask_no_previous_condition, "y_pred"] = np.array(
            [
                str(c_)
                for c_ in np.random.randint(
                    low=1, high=22, size=mask_no_previous_condition.sum()
                )
            ]
        )
        naive_predictions.loc[
            mask_no_previous_condition, "y_pred"
        ] = naive_predictions.loc[mask_no_previous_condition, "y_pred"].apply(
            lambda x: np.array([x])
        )
        return naive_predictions["y_pred"]

    def predict_proba(self, X, mlb: MultiLabelBinarizer = None):
        """
        Transform a multilabel binary predictions of shape (n_patients x n_classes)
        into a list of probability scores (compatible with
        MultiOutputClassifier.predict_proba output), ie. a list of length n_classes
        with elements of shape (n_patients x 2).
        """
        y_pred = self.predict(X)
        if mlb is None:
            # TODO: Not sure this part makes sense when passing other things than
            # binary classification.
            y_prob = np.zeros((y_pred.shape[0], 2))
            y_prob[y_pred == 1, 1] = 1
            y_prob[y_pred == 0, 0] = 1
        else:
            y_pred_binary = mlb.transform(y_pred)
            c_ = 0
            y_prob = []
            for c_ in range(y_pred_binary.shape[1]):
                y_pred_c = y_pred_binary[:, c_]
                y_prob_c_ = np.zeros((y_pred_binary.shape[0], 2))
                y_prob_c_[y_pred_c == 1, 1] = 1
                y_prob_c_[y_pred_c == 0, 0] = 1
                y_prob.append(y_prob_c_)
        return y_prob
