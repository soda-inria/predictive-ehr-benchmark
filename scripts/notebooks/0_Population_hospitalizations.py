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
import databricks.koalas as ks
import pyspark.sql.functions as func
from eds_scikit.io import HiveData
from eds_scikit import improve_performances
from eds_scikit.resources import registry

from copy import copy

from loguru import logger
import seaborn as sns

from medem.preprocessing.quality import time_plot
from medem.utils import save_figure_to_folders
from medem.constants import DIR2DOCS_EXPLORATION

# Use to accelerate autocomplete (seems crucial), should work permanently since it is added to ~/.ipython/ipython_config.py : does not.... don't know why
# %config Completer.use_jedi = False
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.max_columns", 100)
# %load_ext autoreload
# %autoreload 2
#dir2figure = Path(f"{CSE_PROJECT_DB}/timeplots")
#dir2figure.mkdir(exist_ok=True, parents=True)

# %%
# connection au système de requêtage
spark, sc, sql = improve_performances()

# %%
database_name = "cse210038_20220921_160214312112"
sql(f"use {database_name}")
tablenames = sql("show tables").toPandas()
tablenames["tableName"].values

# %%
database_name = "cse210038_20220921_160214312112"
sql(f"use {database_name}")
tablenames = sql("show tables").toPandas()
database = HiveData(database_name, tables_to_load={table_:None for table_ in tablenames["tableName"].values})
for table_name_ in tablenames["tableName"].values:
    df = database.__getattr__(table_name_)
    df = df[[col for col in df.columns if not col.endswith("date")]]
    database.__setattr__(table_name_, df)

# %%
print(database.available_tables)

# %%
person = pd.DataFrame(database.person)

# %% [markdown]
# ## Selection of the Population 
#
# Our base population will be patients with at least two hospitalizations. 

# %%
hospital_visits = database.visit_occurrence.loc[database.visit_occurrence["visit_source_value"].isin(["hospitalisés", "hospitalisation incomplète"])].to_pandas()
hospital_visits.head()

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
hospitalization_types = ["hospitalisés", "hospitalisation incomplète"]
for visit_source_ in hospitalization_types:
    hospital_visits_by_type = hospital_visits.loc[hospital_visits["visit_source_value"]==visit_source_]
    df_, ax = time_plot(hospital_visits_by_type, colname_datetime="visit_start_datetime", ax=ax, label=visit_source_)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set(xlim=("2010", "2023"), xlabel="Year", ylabel="Nb hospitalization")
plt.legend()
save_figure_to_folders("exploration/hospitalization_time_plot")

# %% [markdown]
# # What is the repartition per hospital ? 
#
# ## Check of SI stationnarity
# Compute a ratio of hospitalization (pulled incomplete and complete ones) per hospital activity. 
# The hospital activity is openly available in the SAE and the ratio should not vary differently for a given hospital following time. 
# This is a **check of stationnarity** of the SI in the different hospitals.

# %%
from dateutil.parser import parse
from loguru import logger

# limit the field of study
study_start = parse("2012-01-01")
study_end = parse("2023-06-01")

hospital_visits_study = hospital_visits.loc[
    (hospital_visits["visit_start_datetime"] >= study_start)
    & (hospital_visits["visit_start_datetime"] <= study_end)
]
n_patients = hospital_visits_study.person_id.nunique() 
n_hospitalizations = hospital_visits_study["visit_occurrence_id"].nunique()
logger.info(f"{n_patients} patients, {n_hospitalizations} hospitalizations")

# %%
from eds_scikit.resources import registry
from medem.constants import DIR2RESOURCES
dir2hierarchy_w_hospital_name = DIR2RESOURCES/"care_site_hierarchy_w_hospital_name.csv"
if not dir2hierarchy_w_hospital_name.exists():
    care_site_hierarchy = registry.get("data", function_name="get_care_site_hierarchy")()
    care_site_hierarchy_w_hospital = (
        care_site_hierarchy.merge(
            database.care_site.loc[
                database.care_site["care_site_type_source_value"] == "Hôpital",
                ["care_site_name", "care_site_id"]
            ].to_pandas(),
            left_on="Hôpital",
            right_on="care_site_id", 
            how="inner"
        )
        .drop("Hôpital", axis=1)
        .rename(columns={"care_site_id_x":"care_site_id","care_site_id_y":"hospital_id", "care_site_name":"hospital_name"})
    )
    care_site_hierarchy_w_hospital.to_csv(dir2hierarchy_w_hospital_name,index=False)
else:
    care_site_hierarchy_w_hospital = pd.read_csv(dir2hierarchy_w_hospital_name)
care_site_hierarchy_w_hospital.head()

# %%
hospital_visits_study_w_hospital = hospital_visits_study.merge(
    care_site_hierarchy_w_hospital[["care_site_id", "hospital_name"]],
    on="care_site_id"
)

# %%
hospital_visits_study_w_hospital.head()

# %%
from medem.preprocessing.quality import temporal_heatmap
n_visits_per_hospital, ax = temporal_heatmap(
    event=hospital_visits_study_w_hospital,
    colname_datetime="visit_start_datetime",
    colname_of_interest="hospital_name",
    count_norm="linear"
)
ax.annotate("Nb hospitalizations", xy=(0.95, 1.01), xycoords="axes fraction", 
            fontsize=20, ha="left")
ax.set_xlabel("Year", fontsize=20)
ax.set_ylabel("Hospital", fontsize=20)
ax.tick_params(axis="x", labelsize=20, rotation=50)
ax.tick_params(axis="y", labelsize=16)
save_figure_to_folders("exploration/n_hospitalizations_per_hospital")

# %% [markdown]
# # What is the proprotion of patients having hospitalizations in different hospitals ? 
#
# I restrict to the included population: aged more than 18 at first hospitalization and only during the study period.

# %% jupyter={"outputs_hidden": true}
from medem.utils import force_datetime
from medem.preprocessing.hospitalization_cohort import select_population 

from dateutil.parser import parse

person = database.person.to_pandas()
hospital_visits = database.visit_occurrence.loc[database.visit_occurrence["visit_source_value"].isin(["hospitalisés", "hospitalisation incomplète"])].to_pandas()
hospital_visits = force_datetime(hospital_visits, "visit_start_datetime")

study_start = parse("2017-01-01")
study_end = parse("2022-06-01")
min_age_at_admission = 18
inclusion_criteria = select_population(
    database=database,
    study_start=study_start,
    study_end=study_end,
    min_age_at_admission=min_age_at_admission
)

# %%
inclusion_criteria

# %%

hospital_visits_study_koalas = hospital_visits_koalas.loc[
(hospital_visits_koalas["visit_start_datetime"] >= study_start)
& (hospital_visits_koalas["visit_start_datetime"] <= study_end)
]

n_hospits_per_patients = hospital_visits_study.groupby("person_id")[
    "visit_occurrence_id"
].nunique()

q_095 = n_hospits_per_patients.quantile(0.95)
mask_too_many_hospitalization = n_hospits_per_patients >= q_095

# %%
from eds_scikit.resources import 

# %%
inclusion_criteria["person_id"].count()

# %%
hospitalization_in_population = (
    hospital_visits_study
        .merge(inclusion_criteria[["person_id"]].drop_duplicates(), on="person_id", how="inner")
        .merge(care_site_hierarchy_w_hospital[["care_site_id", "hospital_name"]],
    on="care_site_id")
)
hospitalization_in_population.shape

# %%
n_unique_hospitals_by_patient = hospitalization_in_population.groupby("person_id").agg(**{
    "nb_unique_hospitals": pd.NamedAgg("hospital_name", lambda x: len(np.unique(x)))
})
n_unique_hospitals_by_patient["nb_unique_hospitals"].value_counts()

# %% [markdown]
# # Exploration for selection on hospitalization 

# %% [markdown]
# I have to detect hospitalizations in the data. The hospitalization type information is contained in the `visit_source_value` column of the omop data table. 
# This data is rarely missing. 

# %%
mask_visit_missing = database.visit_occurrence["visit_source_value"].isna()
n_missings_visit_source_value = mask_visit_missing.sum()
n_visits = database.visit_occurrence.shape[0]
logger.info(f"Ratio of missing rows {n_missings_visit_source_value/n_visits:.4f}={n_missings_visit_source_value}/{n_visits}")

# %%
n_visit_by_source = database.visit_occurrence["visit_source_value"].value_counts().to_pandas()
print(n_visit_by_source.to_markdown())

# %% [markdown]
# Theo in the diabete database has a lot of supplementary visits with missing `source_value` but I have almost none missing. However, we do not know why they are missing. Are these visits describing another type of visits or do they belong to one of the 4 classical types ?
#
# We could compare the ratio per patient of the visits on a homogenous phenotype (eg. cim10+biology diabete phenotype?) to see if we have the same. 

# %% [markdown]
# ### What is the link between visit_occurrence and visit_details
#
# - For hospitalization, does one line in visit_occurrence corresponds to one hospitalization ? With multiple ufr associated to each hospitalization, I would be tempted to say yes. However, there is no good way to be sure. 
# - For hospitalization, are there multiple lines in visit_details (corresponding probably to multiple units) ? Yes, the median is at 4 for both incomplete and complete hospitalizations.
# - For consultation externe or urgences, how many lines are they in the visit_details table ? 2 exactly: one PASS and one MOUV for each visit. The MOUV lines are systematically rattached to the PASS line through the visit_detail_parent_id column. 
#   - For the consultation externes, there is no ufr rattached to the MOUV lines. To the PASS lines, there are a non-negligeable part of the lines that are linked only at the hospital level (wo caracterization of the service : more than 15%). 
#   - For the urgence, there is a difference in care_site for the two lines (PASS/MOUV), but wo a clear logic appearing to me.
#
# #### Description of the data: 
#
# - Visit:
# Nb lines 1013046, 
# Nb unique visit_occurence_id 1013046
#
# - Visit detail:
# Nb lines 2648877,
# Nb unique visit_detail_id 2648877,
# Nb unique visit_occurence_id 1012884
#
# There is almost always at least one visit_detail by visit. 

# %%
visits = database.visit_occurrence[["visit_occurrence_id", "person_id", "visit_start_datetime", "visit_end_datetime", "visit_source_value", "care_site_id"]].to_pandas().rename(columns={"care_site_id":"visit_care_site_id"})
cols_visit_details = ["visit_occurrence_id", "visit_detail_id", "person_id", "visit_detail_start_datetime", "visit_detail_end_datetime"]
visit_details = database.visit_detail.to_pandas()

# %%
print("- Visit")
print("Nb lines", visits.shape[0])
print("Nb unique visit_occurence_id", visits["visit_occurrence_id"].nunique())

print("\n- Visit detail")
print("Nb lines",visit_details.shape[0])
print("Nb unique visit_detail_id", visit_details["visit_detail_id"].nunique())
print("Nb unique visit_occurence_id", visit_details["visit_occurrence_id"].nunique())

# %%
visit_w_details = visits.merge(visit_details, on=["person_id", "visit_occurrence_id"], how="inner")
visit_w_details.shape
print(visit_w_details["visit_occurrence_id"].nunique())

# %%
label_count = "Nb distinct visit_detail_id"
n_visit_details_by_visit = (
    visit_w_details
    .groupby(["visit_occurrence_id", "visit_source_value"])
    .agg(**{label_count: pd.NamedAgg("visit_detail_id", lambda x: len(np.unique(x)))}).reset_index()
)

# %%
n_visit_details_by_visit.groupby("visit_source_value")[label_count].describe(percentiles=np.array([10, 25, 50, 75, 90, 95, 99])/100)

# %%
g = sns.boxplot(data=n_visit_details_by_visit, x="visit_source_value", y=label_count)
g.set(ylim=(0, 12))
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
save_figure_to_folders(DIR2RESOURCES)

# %% [markdown]
# #### Why exactly two visit_details for urgence and consultation externe ? 

# %%
other_than_hospitalisations = visit_w_details.loc[visit_w_details["visit_source_value"].isin(["urgence", "consultation externe"])].sort_values(["visit_occurrence_id"])
other_than_hospitalisations["visit_detail_type_source_value"].value_counts()

# %%
care_sites = database.care_site.to_pandas()

# %%
# mouv only 
hospitalization_type_ = "urgence"
mask_hospit_type_ = other_than_hospitalisations["visit_source_value"] == hospitalization_type_
mask_type_source_value = other_than_hospitalisations["visit_detail_type_source_value"]=="MOUV"
mouv_only = other_than_hospitalisations[mask_type_source_value&mask_hospit_type_]
care_site_col = "care_site_id"
(
    mouv_only[care_site_col].value_counts(normalize=True)
    .to_frame().reset_index().rename(columns={"index":"care_site_id", care_site_col:"count"})
    .merge(care_sites[["care_site_id", "care_site_name"]], on="care_site_id", how="inner").head(40)
)

# %% [markdown]
# Do we have more time information in one or the other of the doublon PASS/MOUV ? Not at all.
# - urgence : 0 non consistent start time between PASS and MOUV, 539 for end time
# - consultation externe : 0 non consistent start time between PASS and MOUV, 0 for end time.

# %%
hospitalization_type_ = "consultation externe"
mask_hospit_type_ = other_than_hospitalisations["visit_source_value"] == hospitalization_type_
single_visit_type = other_than_hospitalisations[mask_hospit_type_]
single_visit_type_deduplicated = single_visit_type.groupby(["visit_detail_type_source_value", "visit_occurrence_id"]).agg("first").reset_index()
print(single_visit_type_deduplicated.shape)

# %%
pivoted_single_visit = single_visit_type_deduplicated.pivot(index="visit_occurrence_id", columns="visit_detail_type_source_value", values=["visit_detail_start_datetime", "visit_detail_end_datetime"])

# %%
mask_non_consistent_start = pivoted_single_visit["visit_detail_start_datetime"]["MOUV"] != pivoted_single_visit["visit_detail_start_datetime"]["PASS"] 
mask_non_consistent_end = (pivoted_single_visit["visit_detail_end_datetime"]["MOUV"] != pivoted_single_visit["visit_detail_end_datetime"]["PASS"] ) & (~pivoted_single_visit["visit_detail_end_datetime"]["PASS"].isna() | ~pivoted_single_visit["visit_detail_end_datetime"]["MOUV"].isna())
print(mask_non_consistent_start.sum())
print(mask_non_consistent_end.sum())

# %%
pivoted_single_visit[mask_non_consistent_end]

# %% [markdown]
# # GHM methods [deprecated]

# %%
database.visit_detail["visit_detail_type_source_value"].value_counts()

# %%
rss = database.visit_detail.loc[database.visit_detail["visit_detail_type_source_value"]=="RSS"].to_pandas()
rum = database.visit_detail.loc[database.visit_detail["visit_detail_type_source_value"]=="RUM"].to_pandas()
ghm = database.cost.to_pandas()
print(f"rss counts: {rss.shape}")
print(f"rum counts: {rum.shape}")
print(f"ghm counts: {ghm.shape}")

# %%
# Link between ghm and rum/rss can be made trough cost_even_id. However there are several ghm for one rss or one rum which should not be the case.
rss_ghm = rss.merge(ghm, right_on="cost_event_id", left_on="visit_occurrence_id")
rum_ghm = rum.merge(ghm, right_on="cost_event_id", left_on="visit_occurrence_id")

# %%
