# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: event2vec-py310
#     language: python
#     name: event2vec-py310
# ---

# %%
from pathlib import Path
import pandas as pd
#import numpy as np
import re
import polars as pl

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
    database = HiveData(database_name)
    print(database.available_tables)

# %%
database_name = "cse210038_20220921_160214312112"
cse_path = f"hdfs://bbsedsi/apps/hive/warehouse/bigdata/omop_exports_prod/hive/{database_name}.db/"

import pyarrow.dataset as ds

def scan_from_hdfs(table, cse_path):
    table_path = cse_path + table
    return pl.scan_ds(ds.dataset(table_path))


# %%
person = scan_from_hdfs("person", cse_path)

# %%
person = person.drop("provider_id")

# %%

omop_tables = [
    "care_site",
    "concept",
    "concept_relationship",
    "condition_occurrence",
    "cost",
    "drug_exposure_administration",
    "drug_exposure_prescription",
    "fact_relationship",
    "measurement",
    "note_deid",
    "person",
    "procedure_occurrence",
    "visit_detail",
    "visit_detail_old",
    "visit_occurrence",
    "visit_occurrence_old",
    "vocabulary",
]


def scan_from_hdfs(table, cse_path):
    table_path = cse_path + table
    return pl.scan_ds(ds.dataset(table_path))


class PolarsData:
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.cse_path = f"hdfs://bbsedsi/apps/hive/warehouse/bigdata/omop_exports_prod/hive/{database_name}.db/"
        self.available_tables = omop_tables

        for table_name in self.available_tables:
            setattr(self, table_name, self.load_table(table_name))

    def load_table(self, table_name: str):
        return scan_from_hdfs(table_name, self.cse_path)



# %%
database = PolarsData(database_name=database_name)

# %%
database.visit_occurrence.limit()

# %%
person_sample = person.limit(100)
person_df = person_sample.collect()

# %%
tt = pd.DataFrame(person_df, columns=person_df.columns)
person_df[["person_id"]].lazy()

# %%
df = pl.concat([person, person])

# %%
visit_sample = database.visit_occurrence.join(
    person_sample, on="person_id"
)
tt = visit_sample.collect()

# %%
tt

# %% [markdown]
# # Load cohort

# %%

# %%
config = CONFIG_PROGNOSIS_COHORT
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
# ## What is the delta of time between the target visit and the inclusion event (first stay in the period)

# %%
delta_target_to_first_stay = (targets[COLNAME_FOLLOWUP_START] - targets[COLNAME_INCLUSION_EVENT_START]).dt.days
np.quantile(delta_target_to_first_stay, q=[0.1, 0.25, 0.5, 0.75, 0.9])

# %%
ax = delta_target_to_first_stay.hist(bins=50)
ax.set(xlim=(-1, 1500))

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
