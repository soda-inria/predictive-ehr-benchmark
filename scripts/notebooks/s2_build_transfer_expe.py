# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: event2vec
#     language: python
#     name: event2vec
# ---

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import re

from eds_scikit.io import HiveData
from eds_scikit import improve_performances
from databricks import koalas as ks

from dateutil.parser import parse

from medem.utils import save_figure_to_folders, add_age
from medem.constants import *

from medem.experiences.configurations import CONFIG_PROGNOSIS_COHORT, CONFIG_LOS_COHORT, cohort_configuration_to_str, FEATURIZER_SNDS, PATH2SNDS_EMBEDDINGS
from medem.experiences.utils import combine_featurizer_estimator
from medem.experiences.cohort import EventCohort
from medem.experiences.pipelines import OneHotEvent, Event2vecFeaturizer, build_vocabulary, Event2vecPretrained, get_feature_sparsity, restrict_to_vocabulary
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer 

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2

# %%
LOAD_DATA = False
if LOAD_DATA:
    # connection au système de requêtage
    spark, sc, sql = improve_performances()
    database_name = "cse210038_20220921_160214312112"
    sql(f"use {database_name}")
    tablenames = sql("show tables").toPandas()
    database = HiveData(database_name, tables_to_load={table_:None for table_ in tablenames["tableName"].values})
    for table_name_ in tablenames["tableName"].values:
        df = database.__getattr__(table_name_)
        df = df[[col for col in df.columns if not col.endswith("date")]]
        database.__setattr__(table_name_, df)
    print(database.available_tables)

# %% [markdown]
# # Load cohort

# %%
config = CONFIG_LOS_COHORT
cohort_name = cohort_configuration_to_str(config)

dir2report_imgs = DIR2DOCS_COHORT / cohort_name
dir2report_imgs.mkdir(exist_ok=True)
dir2cohort = DIR2DATA / cohort_name
event_cohort = EventCohort(folder=dir2cohort)
print(event_cohort.person.shape)
print(event_cohort.event.shape)
event_cohort.event.head()
targets = event_cohort.person

# %% [markdown]
# ### check the split for the transfer

# %%
split_hospitals

# %%
split_hospitals = pd.read_parquet(dir2cohort / "hospital_split.parquet")
train_person = event_cohort.person.merge(
    split_hospitals.loc[split_hospitals["dataset"] == "train"],
    on="person_id",
    how="inner",
)
train_event = event_cohort.event.merge(
    train_person[COLNAME_PERSON],
    on=COLNAME_PERSON,
    how="inner",
)
test_person = event_cohort.person.merge(
    split_hospitals.loc[split_hospitals["dataset"] == "external_test"],
    on="person_id",
    how="inner",
)
test_event = event_cohort.event.merge(
    test_person[COLNAME_PERSON],
    on=COLNAME_PERSON,
    how="inner",
)
print(event_cohort.person.shape)
print(test_event.shape, test_person.shape)
print(train_event.shape, train_person.shape)

# %% [markdown]
# ### check prevalences

# %%
from medem.preprocessing.tasks import get_prognosis_prevalence
prevalences = get_prognosis_prevalence(targets[COLNAME_OUTCOME])
prevalences.

# %%
# forces 
mlb = MultiLabelBinarizer()
mlb.fit(targets[COLNAME_OUTCOME])
y = mlb.transform(targets[COLNAME_OUTCOME])

# %%
prevalences = get_prognosis_prevalence(y, classes=mlb.classes_)
prevalences
#prevalences.to_dict(orient="records")[0]

# %% [markdown]
# # test the prognosis task

# %%
from medem.experiences.configurations import ESTIMATORS_TASK_LOS, FEATURIZERS_TASK_LOS

featurizer_config=FEATURIZERS_TASK_LOS[3]
estimator_config=ESTIMATORS_TASK_LOS[1]

# setup hps for multilabel
new_estimator_kwargs = {}
for k, v in estimator_config["estimator_kwargs"].items():
    new_estimator_kwargs[f'estimator__{k}'] = v
estimator_config["estimator_kwargs"] = new_estimator_kwargs
    
featurizer_estimator_config = combine_featurizer_estimator(
    event_featurizer=featurizer_config, 
    estimator_pipeline=estimator_config, 
    #preprocessor=preprocessor
)

featurizer = featurizer_estimator_config["featurizer"]
estimator = featurizer_estimator_config["estimator"]
estimator = MultiOutputClassifier(estimator)

study_vocabulary = build_vocabulary(
        event=event_cohort.event,
        n_min_events=10,
)

print(featurizer)
print(estimator)

# %%
X, person_aligned = featurizer.fit_transform(
        event=event_cohort.event,
        person=event_cohort.person,
        vocabulary=study_vocabulary,
    )

# %%
outcome_classes = targets[COLNAME_OUTCOME].explode().unique()
mlb = MultiLabelBinarizer(classes=outcome_classes)
mlb.fit(targets[COLNAME_OUTCOME])
y = mlb.transform(targets[COLNAME_OUTCOME])

# %%
pipeline = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=featurizer_estimator_config["estimator_kwargs"],
            scoring="roc_auc",
            n_iter=3,
            n_jobs=3,
            random_state=0,
        )
pipeline

# %% jupyter={"outputs_hidden": true}
pipeline.fit(X, y)
# https://scikit-learn.org/stable/modules/multiclass.html

# %%
from medem.experiences.utils import get_scores
hat_y_proba = pipeline.predict_proba(X)
scores = {}
for i, single_target in enumerate(mlb.classes):
    scores[single_target] = get_scores(y[:, i], hat_y_proba[i][:, 1])

# %%
