# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: codia_large
#     language: python
#     name: codia_large
# ---

# %%
from eds_scikit.io import HiveData

from eds_scikit.period.stays import merge_visits
from eds_scikit import improve_performances
import datetime
from pathlib import Path

spark, sc, sql = improve_performances()

# %%
data = HiveData("cse_210037_20221028", spark_session=spark, database_type="I2B2")
visit_occurrence_raw = data.visit_occurrence

# %%
visit_types = data.visit_occurrence.groupby("visit_source_value").count().toPandas()
visit_types

# %%
visit_occurrence = merge_visits(
    visit_occurrence_raw, 
    max_timedelta=datetime.timedelta(days=2),
    merge_different_hospitals=True,
    merge_different_source_values=["hospitalisation incomplète", "urgence", "hospitalisés"]
)

# %%
dir2mace_cohort = Path("file:///export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace_cohort")

visit_occurrence.to_parquet(str(dir2mace_cohort/"visit_occurrence_merged"))

# %%
