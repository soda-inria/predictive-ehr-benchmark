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

from medem.experiences.configurations import *
from medem.experiences.cohort import EventCohort
from medem.experiences.pipelines import (
    OneHotEvent,
    Event2vecFeaturizer,
    build_vocabulary,
    Event2vecPretrained,
    get_feature_sparsity,
    restrict_to_vocabulary,
)
from medem.experiences.utils import combine_featurizer_estimator

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

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
    database = HiveData(
        database_name,
        tables_to_load={
            table_: None for table_ in tablenames["tableName"].values
        },
    )
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

# %%
# extract vocabulary with at least 10 codes:

vocabulary = build_vocabulary(event_cohort.event)

# %%
event_cohort.event.loc[
    event_cohort.event["event_source_type_concept_id"] == "condition_occurrence"
]["event_source_concept_id"].map(lambda x: len(x)).value_counts()

vocabulary_provenance = pd.DataFrame(
    {"event_source_concept_id": vocabulary}
).merge(
    event_cohort.event[
        ["event_source_concept_id", "event_source_type_concept_id"]
    ].drop_duplicates()
)
print(len(vocabulary))
print(len(vocabulary_provenance))
vocabulary_provenance.to_csv(
    DIR2RESOURCES / ("vocabulary_10p__" + cohort_name + ".csv")
)

# %% [markdown]
# # static transformers

# %%
from medem.experiences.features import get_date_details

static_features = get_date_details(
    event_cohort.person, colname_datetime=COLNAME_INCLUSION_EVENT_START
).drop("inclusion_time_of_day", axis=1)
static_features = add_age(
    df=static_features,
    ref_datetime=COLNAME_INCLUSION_EVENT_START,
)
static_features.columns

event_cohort.person = static_features

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from medem.experiences.configurations import (
    STATIC_FEATURES_DICT,
    CONFIG_EVENT2VEC_TEST,
)

config_experience = CONFIG_EVENT2VEC_TEST
static_features_config = config_experience.get("colname_demographics", None)
colname_demographics = [
    *static_features_config["categorical"],
    *static_features_config["numerical"],
]
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
categorical_columns = static_features_config["categorical"]
numerical_columns = static_features_config["numerical"]

categorical_ix = np.arange(len(categorical_columns))
numerical_ix = np.arange(
    start=len(categorical_columns),
    stop=len(categorical_columns) + len(numerical_columns),
)
"""
# ugly but necessary for the ColumnTransformer to work on sparse matrices
def _get_sparse_cat_col(X):
    accumulator = []
    for col in categorical_ix:
        accumulator.append(X.getcol(col).toarray())
    return np.hstack(accumulator)

def _get_sparse_num_col(X):
    accumulator = []
    for col in numerical_ix:
        accumulator.append(X.getcol(col).toarray())
    return np.hstack(accumulator)
"""

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_ix),
        ("standard_scaler", numerical_preprocessor, numerical_ix),
    ],
    remainder="passthrough",
)

# %%
featurizer_config = {
    "featurizer_name": FEATURIZER_CUI2VEC,
    "featurizer": Event2vecPretrained(
        path2embedding=PATH2SNDS_EMBEDDINGS,
        n_min_events=N_MIN_EVENTS,
    ),
}
estimator_config = {
    "estimator_name": "ridge",
    "estimator": [
        LogisticRegression(class_weight="balanced", n_jobs=-1),
    ],
    "estimator_kwargs": {"logisticregression__C": [1e-3, 1e-2, 1e-1, 5e-1, 1]},
}
featurizer, estimator = combine_featurizer_estimator(
    event_featurizer=featurizer_config["featurizer"],
    estimator_pipeline=estimator_config["estimator"],
)

print(featurizer)
print(estimator)

# %%
featurizer.embedding

# %%
X = featurizer.fit_transform(X=event_cohort.event, y=event_cohort.person)
y = event_cohort.person[COLNAME_OUTCOME].values

# %%
print(len(featurizer.vocabulary_))
all_vocab = build_vocabulary(event_cohort.event)
print(len(all_vocab))

# %%
# what codes are not mapped for cui2vec, by vocabulary
mapped_vocabulary = (
    event_cohort.event[[COLNAME_SOURCE_CODE, COLNAME_SOURCE_TYPE]]
    .drop_duplicates()
    .merge(pd.DataFrame({COLNAME_SOURCE_CODE: all_vocab}))
)
mapped_vocabulary_cui2vec = mapped_vocabulary.merge(
    pd.DataFrame({COLNAME_SOURCE_CODE: featurizer.vocabulary_, "cui2vec": 1}),
    how="left",
).fillna(0)

# %%
print(
    mapped_vocabulary_cui2vec.groupby("event_source_type_concept_id")["cui2vec"]
    .mean()
    .round(2)
    .to_markdown()
)

# %%
mask_unmapped_codes = mapped_vocabulary_cui2vec["cui2vec"] == 0
mapped_vocabulary_cui2vec.loc[
    mask_unmapped_codes
    & (mapped_vocabulary_cui2vec[COLNAME_SOURCE_TYPE] != "procedure_occurrence")
]

# %%
import scipy.sparse as sp
from sklearn.preprocessing import OrdinalEncoder

static_features = np.hstack(
    [
        OrdinalEncoder().fit_transform(
            event_cohort.person[categorical_columns]
        ),
        event_cohort.person[numerical_columns],
    ]
)
# that seems complicated to stack dense and
X_full = sp.hstack([static_features, X])

# %%
y_pred = estimator_pipeline.fit(X_full.toarray(), y)

# %%
estimator_pipeline

# %% [markdown]
# ## inpsect outcome

# %%

# %%
event_cohort.person[COLNAME_LOS_CATEGORY].value_counts()

# %%
nan_outcome = event_cohort.person[COLNAME_LOS] == 0
event_cohort.person[nan_outcome]

# %% [markdown]
# # inspect vocabulary

# %%
vocabulary = build_vocabulary(event_cohort.event, n_min_events=10)
print(len(vocabulary))
restricted_events = restrict_to_vocabulary(
    event_cohort.event, vocabulary=vocabulary
)
print(restricted_events["event_source_type_concept_id"].value_counts())
print(
    "Number of patients with event in vocabulary:",
    restricted_events["person_id"].nunique(),
)
restricted_patients = event_cohort.person.merge(
    restricted_events[["person_id"]].drop_duplicates(),
    on="person_id",
    how="inner",
)
targets_w_age = add_age(restricted_patients, ref_datetime="followup_start")
print((targets_w_age["gender_source_value"] == "f").mean())
targets_w_age[[COLNAME_OUTCOME, "age_to_followup_start"]].describe()

# %%
snds_featurizer = Event2vecPretrained(
    path2embedding=PATH2SNDS_EMBEDDINGS,
    colname_code=COLNAME_SOURCE_CODE,
)
snds_featurizer.embedding = pd.read_parquet(snds_featurizer.path2embedding)
print(len(snds_featurizer.embedding.columns))
common_vocabulary = set(vocabulary).intersection(
    set(snds_featurizer.embedding.columns)
)
print(len(common_vocabulary))

# %%
###
from medem.preprocessing.hospitalization_cohort import create_outcome

target = create_outcome(
    database=database,
    inclusion_criteria=event_cohort.person[
        ["person_id", COLNAME_FOLLOWUP_START, "inclusion_start"]
    ],
    horizon_in_days=7,
    deceased="include",
)
###

# %%
n_min_events = 10
vocabulary = _build_vocabulary(
    event=event_cohort.event,
    colname_code=COLNAME_SOURCE_CODE,
    n_min_events=n_min_events,
)

event_restricted_to_vocabulary = event_cohort.event.merge(
    pd.DataFrame({COLNAME_SOURCE_CODE: vocabulary}),
    on=COLNAME_SOURCE_CODE,
    how="inner",
)

# %%
event_restricted_to_vocabulary.shape

# %% [markdown]
# # Test from outside embeddings

# %%

# %%
X, y = pretrained_featurizer.fit_transform(
    event=event_cohort.event,
    person=event_cohort.person,
)

# %%
from sklearn import preprocessing

dem["gender_source_value"] = preprocessing.LabelEncoder().fit_transform(
    dem["gender_source_value"]
)

# %%
dem

# %%
from medem.utils import add_age

dem = add_age(event_cohort.person, ref_datetime=COLNAME_INCLUSION_START)


# %%
n = 1000
X_sub, y_sub = X[:n], y[:n]

# %%
from medem.experiences.configurations import (
    CONFIG_EVENT2VEC_PERFORMANCES,
    FEATURIZER_COUNT,
    DEFAULT_ESTIMATORS,
)

estimator_config = DEFAULT_ESTIMATORS[0]
estimator_pipeline = make_pipeline(LogisticRegression(class_weight="balanced"))

gridsearch = RandomizedSearchCV(
    estimator=estimator_pipeline,
    param_distributions=estimator_config["estimator_kwargs"],
    scoring="roc_auc",
    n_iter=3,
    n_jobs=-1,
    random_state=0,
)

# %%
gridsearch.fit(X_sub, y_sub)

# %%
snds_event_featurizer.vocabulary = list(raw_intersection_w_snds)
X, y = snds_event_featurizer.transform(
    event=event_cohort.event,
    person=event_cohort.person,
)

# %%
X.shape

# %% [markdown]
# # Inside embeddings sparsity

# %%
in_domain_featurizer = Event2vecFeaturizer(
    colname_code=COLNAME_SOURCE_CODE,
    window_radius_in_days=60,  # should cover twice the horizon
    window_orientation="center",
    backend="pandas",
    d=30,
)
X_in, y_in = in_domain_featurizer.fit_transform(
    event=event_cohort.event, person=event_cohort.person
)
get_feature_sparsity(X_in)

# %% [markdown]
# # Test Event2vecFeaturizer

# %%
from joblib import dump, load

# %%
dump(featurizer, featurizer.output_dir / "event2vec.joblib")

# %%
featurizer2 = load(featurizer.output_dir / "event2vec.joblib")

# %% [markdown]
# # Test only event2vec

# %%
from event2vec.svd_ppmi import build_cooccurrence_matrix_pd

cooccurrence_matrix, event_count, label2ix = build_cooccurrence_matrix_pd(
    events=restricted_events[:100000],
    output_dir=dir2cohort,
    radius_in_days=30,
    window_orientation="center",
)
cooccurrence_matrix.todense()

# %%
# %%time
embeddings = event2vec(
    events=restricted_events[:200000],
    output_dir=dir2cohort,
    colname_concept=COLNAME_SOURCE_CODE,
    window_radius_in_days=30,
    window_orientation="center",
    backend="pandas",
    d=50,
)

# %% [markdown]
# # One hot encoding

# %%
one_hot_encoder = OneHotEvent(sparse=False, decay_half_life_in_days=[0, 7])
one_hot_encoder.fit(event_cohort.event)
X, statics = one_hot_encoder.transform(
    X=event_cohort.event, y=event_cohort.person
)

# %%
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from medem.experiences.configurations import (
    ESTIMATOR_LIST_TEST,
    CONFIG_EVENT2VEC_TEST,
    make_featurizer_estimator_config,
    FEATURIZER_LIST_TEST,
)

config_experience = CONFIG_EVENT2VEC_TEST
estimator_config = make_featurizer_estimator_config(
    FEATURIZER_LIST_TEST[0], ESTIMATOR_LIST_TEST[1]
)
estimator_pipeline = HistGradientBoostingClassifier()
pipeline = RandomizedSearchCV(
    estimator=make_pipeline(*estimator_pipeline),
    param_distributions=estimator_config["estimator_kwargs"],
    scoring=config_experience["randomsearch_scoring"],
    n_iter=config_experience["randomsearch_n_iter"],
    n_jobs=-1,
    random_state=config_experience["randomsearch_rs"],
)

# %%
# needs a debug since it takes too much time.
n_patients = int(X.shape[0] / 8)
X_subset = X[:n_patients]
y_subset = statics[:n_patients][COLNAME_OUTCOME].values

print(X_subset.shape, y_subset.shape)
pipeline.fit(X_subset, y_subset)

# %% [markdown]
# # sparsity

# %%
sparsity = 1 - (X != 0).sum().sum() / (X.shape[0] * X.shape[1])
sparsity
