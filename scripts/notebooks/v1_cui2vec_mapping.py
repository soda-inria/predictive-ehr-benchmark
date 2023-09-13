# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from medem.constants import *
from medem.experiences.configurations import PATH2CUI2VEC_FIRST_EMBEDDINGS, PATH2CUI2VEC_MEAN_EMBEDDINGS
import re
# %%
cui2vec = pd.read_parquet(DIR2EMBEDDINGS / "cui2vec" / "cui2vec.parquet").rename(columns={"Unnamed: 0":"CUI"})
cui2vec.head()
# %%
umls_subset = pd.read_parquet(DIR2RESOURCES / "umls_big_subset.parquet")
# %% 
# Only run once to restrict umls to  the vocabularies of interest
# prerequisite : First download the [MRCONSO.RRF file from UMLS]()

#umls_full = pd.read_csv(DIR2RESOURCES / "MRCONSO.RRF", header=None, sep="|")
#umls_full.columns = umls_subset.columns
#umls_full.loc[umls_full["SAB"].isin(
#    ["ICD10", "ATC", "LNC", "SNOMEDCT_US", "ICD10CM", "ICD10PCS"]
#)].to_csv(DIR2RESOURCES / "umls_big_subset.csv", index=False)
#umls_subset = pd.read_csv(DIR2RESOURCES/ 'umls_subset.csv')
#umls_subset = pd.read_csv(DIR2RESOURCES/ 'umls_big_subset.csv', dtype={"SCUI":"object"})
#umls_subset.to_parquet(DIR2RESOURCES / "umls_big_subset.parquet")
# %%
# 14K unmatched cui (TODO: what concepts type are they ?) 
unmatched_cui2vec = set(cui2vec["CUI"].values).difference(umls_subset.CUI)
print(len(unmatched_cui2vec))
# matched cui
cui_to_umls_matched = pd.DataFrame({"CUI": cui2vec["CUI"]}).merge(umls_subset, on='CUI', how='inner')
cui_to_umls_unique = cui_to_umls_matched.drop_duplicates(subset=['CUI'])
cui_to_umls_unique["SAB"].value_counts()
# %% [markdown]
# # Match study concepts to OMOP concepts
# %%
dir2athena = DIR2RESOURCES / "athena_snomed_to_french_nomenclatures" 
athena_concepts = pd.read_csv(dir2athena / "CONCEPT.csv", sep="\t")
athena_concepts = athena_concepts[~athena_concepts["vocabulary_id"].isin(["ICD9CM"])]
study_concept = (
    pd.read_csv(DIR2RESOURCES / "vocabulary_10p__complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3.csv")
    .drop("Unnamed: 0", axis=1)
    .rename(columns={"event_source_concept_id": "concept_code"})
)
vocabulary_mapper = {
    "procedure_occurrence": "CCAM",
    "condition_occurrence": "CIM10",
    "drug_exposure_administration": "ATC",
}
study_concept["event_source_type_concept_id"] = study_concept["event_source_type_concept_id"].map(lambda x: vocabulary_mapper[x])
n_concepts_per_vocabulary = study_concept["event_source_type_concept_id"].value_counts()
print(n_concepts_per_vocabulary)

# remap ICD10 concepts for to be french consistent
mask_athena_icd10 = athena_concepts["vocabulary_id"].isin(["CIM10", "ICD10", "ICD10CM"]) 
athena_concepts.loc[mask_athena_icd10, "concept_code"] = (
    athena_concepts.loc[mask_athena_icd10, "concept_code"]
    .apply(lambda x: re.sub("\.", "", x))
)
# %%
# remap icd10 to be french consistent and troncate at 5 digits since we only have 5 or more in our study.
mask_cui_icd10 = (cui_to_umls_unique["SAB"].isin(["ICD10", "ICD10CM"]))
cui_to_umls_unique.loc[mask_cui_icd10, "CODE"] = cui_to_umls_unique.loc[mask_cui_icd10, "CODE"].apply(lambda x: re.sub("\.", "", x)[:5])
# %% 
# mapping icd10 and icd10cm to cim10 thanks to athena snomed
relations = pd.read_csv(dir2athena / "CONCEPT_RELATIONSHIP.csv", sep="\t")
# %%
cim10_to_all = relations.loc[relations["relationship_id"] == "Maps to"].merge(
    athena_concepts.loc[athena_concepts["vocabulary_id"] == "CIM10", ["concept_code","concept_id"]], left_on="concept_id_1", right_on="concept_id", how="inner"
).drop("concept_id", axis=1).rename(columns={"concept_id_2": "concept_id", "concept_code": "cim10_concept_code"})
cim10_2snomed = cim10_to_all.merge(
    athena_concepts, on="concept_id", how="inner")
# %%
icd10_2snomed= relations.loc[relations["relationship_id"] == "Maps to"].merge(
    athena_concepts.loc[athena_concepts["vocabulary_id"].isin(["ICD10", "ICD10CM"]), ["concept_code","concept_id"]], left_on="concept_id_1", right_on="concept_id", how="inner"
).drop("concept_id", axis=1).rename(columns={"concept_id_2": "concept_id", "concept_code": "icd10_concept_code"}).merge(
    athena_concepts, on="concept_id", how="inner")
# %%
cui2vec_icd10_w_snomed = cui_to_umls_unique.loc[cui_to_umls_unique["SAB"].isin(["ICD10", "ICD10CM"])].merge(
    icd10_2snomed, left_on="CODE", right_on="icd10_concept_code", how="inner"
).merge(cim10_2snomed[["concept_id", "cim10_concept_code"]], on="concept_id", how="inner")
# %% 
unique_mapping_cui2vec_icd10_to_cim10 = cui2vec_icd10_w_snomed[["CUI", "cim10_concept_code"]].drop_duplicates().sort_values("CUI")
unique_mapping_cui2vec_icd10_to_cim10.shape
# %%
# Mapping ccam to snomed ct us
ccam_to_snomed = relations.merge(
    athena_concepts.loc[athena_concepts["vocabulary_id"] == "CCAM", ["concept_code","concept_id"]], left_on="concept_id_1", right_on="concept_id", how="inner"
).drop("concept_id", axis=1).rename(columns={"concept_id_2": "concept_id", "concept_code": "ccam_concept_code"}).merge(
    athena_concepts.loc[athena_concepts["vocabulary_id"]!="CCAM"], on="concept_id", how="inner")
# No mapping for CCAM to another vocabulary in athena....
#  %%
# mapping cui to study_concepts
## for icd10 and icd10cm
mask_study_cim10 = study_concept["event_source_type_concept_id"] == "CIM10"
study2cui2vec_cim10 =( 
    study_concept[mask_study_cim10].merge(
    unique_mapping_cui2vec_icd10_to_cim10, left_on="concept_code", right_on="cim10_concept_code", how="inner"
    ).drop("cim10_concept_code", axis=1)
).drop_duplicates()
## for ATC
mask_study_act = study_concept["event_source_type_concept_id"] == "ATC"
mask_cui_atc = cui_to_umls_unique["SAB"] == "ATC"
study2cui2vec_act =( 
    study_concept[mask_study_act].merge(
    cui_to_umls_unique.loc[mask_cui_atc, ["CUI", "CODE"]], left_on="concept_code", right_on="CODE", how="inner"
    ).drop("CODE", axis=1)
)
## For CCAM. No mapping existing....

## For LNC
# TODO: add mapping for LNC
study2cui2vec = pd.concat([study2cui2vec_cim10, study2cui2vec_act], axis=0)
n_mapped_by_vocabulary = study2cui2vec[["concept_code", "event_source_type_concept_id"]].drop_duplicates()["event_source_type_concept_id"].value_counts()

print("Lost percentage of concepts per vocabulary:")
print((n_concepts_per_vocabulary - n_mapped_by_vocabulary)/n_concepts_per_vocabulary)
print(n_mapped_by_vocabulary.sum())
# %%
study_embeddings = cui2vec.merge(study2cui2vec, on="CUI", how="inner").drop(["CUI", "event_source_type_concept_id"], axis=1)
study_embeddings_grouped_first = study_embeddings.groupby("concept_code").first().transpose()
study_embeddings_grouped_mean = study_embeddings.groupby("concept_code").mean().transpose()
# %%
# if duplicate codes, take the mean of the embeddings
study_embeddings_grouped_first.to_parquet(PATH2CUI2VEC_FIRST_EMBEDDINGS, index=False)
study_embeddings_grouped_mean.to_parquet(PATH2CUI2VEC_MEAN_EMBEDDINGS, index=False)
# save for study
#pd.DataFrame({COLNAME_SOURCE_CODE: study_embeddings_grouped_mean.columns}).to_csv(, index=False)