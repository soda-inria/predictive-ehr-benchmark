# %%
from eds_scikit import improve_performances
from eds_scikit.io import HiveData
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pyspark.sql import functions as F

pd.set_option("display.max_columns", None)

spark, sc, sql = improve_performances()

# %% [markdown] This script is used to copy the data from the HDFS cluster with
# spark to the local machine before being fed to the population scripts.
# It also convert the strict minimum number of columns from i2b2 to omop.
# %%
all_db = sql("SHOW DATABASES").toPandas()
all_db.loc[all_db["databaseName"].str.find("cse_210037") != -1]

db_name = "cse_210037_20221028"
control_cohort = HiveData(f"{db_name}", database_type="I2B2")
# control_cohort.person

all_tables = sql(f"SHOW TABLES FROM {db_name}").toPandas()
all_tables

person = sql(f"select * from {db_name}.i2b2_patient")
person.filter(~F.isnull("age_in_years_num")).show(5)


# %% [markdown]
#
# # I need to reconstruct the age of the patients from the visit table
#


# %%
def add_birth_datetime(i2b2_visit, person, spark):
    "Coarse estimation of birth datetime"
    coarse_first_visit = (
        i2b2_visit.sort("start_date")
        .groupby("patient_num")
        .agg(
            F.first("start_date").cast("string").alias("start_date"),
            F.first("age_visit_in_month_num").alias("age_visit_in_month_num"),
        )
        .toPandas()
    )
    coarse_first_visit["birth_datetime"] = pd.to_datetime(
        coarse_first_visit["start_date"], errors="coerce"
    ) - pd.to_timedelta(
        coarse_first_visit["age_visit_in_month_num"] * 30, unit="day"
    )
    coarse_first_visit = coarse_first_visit.rename(
        columns={"patient_num": "person_id"}
    )
    person_w_age = person.join(
        spark.createDataFrame(
            coarse_first_visit[["person_id", "birth_datetime"]]
        ),
        how="inner",
        on="person_id",
    )
    return person_w_age


recompute_age = False
if recompute_age:
    i2b2_visit = sql(f"select * from {db_name}.i2b2_visit")
    coarse_first_visit = (
        i2b2_visit.sort("start_date")
        .groupby("patient_num")
        .agg(
            F.first("start_date").cast("string").alias("start_date"),
            F.first("age_visit_in_month_num").alias("age_visit_in_month_num"),
        )
        .toPandas()
    )
    coarse_first_visit["birth_datetime"] = pd.to_datetime(
        coarse_first_visit["start_date"], errors="coerce"
    ) - pd.to_timedelta(
        coarse_first_visit["age_visit_in_month_num"] * 30, unit="day"
    )
    coarse_first_visit = coarse_first_visit.rename(
        columns={"patient_num": "person_id"}
    )
    from datetime import datetime

    datetime_ref = datetime.now()

    data_to_plot = coarse_first_visit.copy()
    data_to_plot["age_today"] = (
        datetime_ref - data_to_plot["birth_datetime"]
    ).dt.total_seconds() / (365 * 24 * 3600)
    mask_over_150 = data_to_plot["age_today"] >= 120
    data_to_plot_ = data_to_plot[~mask_over_150]
    ax = (data_to_plot_["age_today"]).plot(kind="hist", bins=20)
    import matplotlib.pyplot as plt

    ax.set(xlim=(-5, 110))


# %% [markdown]
#
# # Getting back omop converted tables

# %%

dir2mace_cohort = Path(
    "file:///export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace_cohort"
)

# takes 10 min to run
# "visit_occurrence", "visit_detail", "person", "concept", "care_site", "condition_occurrence", "procedure_occurrence"
# block might not be useful since we don't have to collect
for table_name in ["person"]:
    print(table_name)
    path2table = dir2mace_cohort / table_name
    table_ = control_cohort.__getattr__(table_name).to_spark()
    """
    for col in table_.columns:
        if col in id_cols:
            table_ = table_.with_column(col, F.col(col).cast())
    """
    if table_name == "person":
        i2b2_visit = sql(f"select * from {db_name}.i2b2_visit")
        table_ = add_birth_datetime(
            i2b2_visit, table_.drop("birth_datetime"), spark
        )
    table_.write.mode("overwrite").parquet(str(path2table))

# %% [markdown]
# # Getting back the drug table

# %%
# To make it usable by my code, I just need the person_id column, then start and source code columns are configurables.
# I will try to distinguish between prescribed and adminstred drugs by creating two different tables before writing to hdfs.
path2table = dir2mace_cohort / "drug_exposure"
drug_tables = sql(f"select * from {db_name}.i2b2_observation_med")
drug_tables_clean = (
    drug_tables.withColumnRenamed("patient_num", "person_id")
    .withColumnRenamed("valueflag_cd", "drug_class_source_value")
    .withColumnRenamed("encounter_num", "visit_occurrence_id")
    .withColumnRenamed("start_date", "start")
    .filter(~F.isnull("drug_class_source_value"))
)

drug_tables_clean.write.mode("overwrite").parquet(str(path2table))

# %% [markdown]
#
# # Polars and hdfs leads to OOM

# %%
import polars as pl
import pyarrow
import pyarrow.dataset as ds
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# %%
USER = os.environ["USER"]
hdfs_url = f"hdfs://bbsedsi/user/{USER}/mace_cohort/"
dir2mace_cohort = Path(
    "/export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace_cohort"
)
dir2mace_cohort.mkdir(exist_ok=True, parents=True)

# %%
table_name = "condition_occurrence"
block_size = 100_000
block_indices = np.arange(block_size, block_size * 2)
table = ds.dataset(hdfs_url + table_name).take(block_indices)
# collected_table = table.slice(offset=block_size, length=block_size).collect()
# collected_table = table.collect()
# table.dtypes

# %%
# does not work beacause the originals are stored in orc format
"""
database_name = "cse_210037_20221028"
hdfs_url = f"hdfs://bbsedsi/apps/hive/warehouse/cse/{database_name}/"
table_name ="i2b2_observation_cim10/"
"""
