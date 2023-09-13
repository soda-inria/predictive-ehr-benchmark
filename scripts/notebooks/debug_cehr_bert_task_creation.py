# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: event2vec
#     language: python
#     name: event2vec
# ---

# %%
import pandas as pd
from eds_scikit import improve_performances
import numpy as np
from pathlib import Path

# %%
#cohort_name = "complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3"
cohort_name = "icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01"
dir2cohort = Path(f"~/Matthieu/medical_embeddings_transfer/data/{cohort_name}")
split = pd.read_parquet(dir2cohort/"dataset_split.parquet")
split["hospital_split"].isna().sum()

# %%
path2sequences_train = dir2cohort/"cehr_bert_finetuning_sequences_train"
path2sequences_test = dir2cohort/"cehr_bert_finetuning_sequences_external_test"

df = pd.read_parquet(path2sequences_train)
df["split_group"].isna().sum()

# %%
df["index_stay_chapters"]

# %%
np.expand_dims(df.age, axis=-1)

# %%
pretrained_seq = pd.read_parquet(dir2cohort/"cehr_bert_sequences/patient_sequence_effective_train")

# %%
subtrained_seq = pd.read_parquet(dir2cohort/"cehr_bert_sequences/patient_sequence_effective_train")
subtrained_seq.head()

# %%

# %%
spark, sc, sql = improve_performances()

# %%
cohort_name = "complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3"
path2person = f"~/Matthieu/medical_embeddings_transfer/data/{cohort_name}/person.parquet"
person = pd.read_parquet(
    path2person
)
print(person.dtypes)
print(person.shape)

# %%
person["number_of_stays"] = person["number_of_stays"].astype("Int64")
print(person.dtypes)
path2newperson= f"~/Matthieu/medical_embeddings_transfer/data/{cohort_name}/person_n.parquet"
person.to_parquet(path2newperson)

# %%
path2newperson= f"file:/export/home/cse210038/Matthieu/medical_embeddings_transfer/data/{cohort_name}/person_n.parquet"
spark_person = spark.read.parquet(path2newperson)
spark_person

# %%
