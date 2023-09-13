# %%
import pandas as pd
import numpy as np

# %% [markdown]
# # Prepare loinc tree for nabm concepts presents in the snds

# %%
nabm2loinc = pd.read_excel(
    "/home/matthieu/Documents/santeFrance/projets/omop_universe/drees_conceptual_mapping/local_counts/athena_data/LOINCFR2nabm_JeuDeValeurs.xlsx",
    sheet_name="Alignement NABM-LOINC FR",
)

nabm_raw_codes = pd.read_csv(
    "/home/matthieu/Documents/santeFrance/projets/hub_santé/healthdatahub_gitlab/schema-snds/nomenclatures/ORAREF/IR_BIO_R.csv",
    sep=";",
)

# %%
selected_columns = {
    "BIO_PRS_IDE": "concept_code",
    "BIO_INF_LIB": "concept_name",
    "BIO_ARB_CHA": "concept_chapter",
    "BIO_ARB_SC1": "sub_chapter_n1",
}
# subchapter 2, 3 and 4 are all 0

# %%
nabm_codes = nabm_raw_codes.loc[:, selected_columns.keys()].rename(
    columns=selected_columns
)

nabm_codes.loc[:, "concept_name"] = nabm_codes.loc[
    :, "concept_name"
].str.lower()


def map_to_int(x):
    if np.isnan(x):
        return -1
    else:
        return 0


nabm_codes.loc[:, "concept_chapter"] = nabm_codes.loc[:, "concept_chapter"].map(
    lambda x: map_to_int(x)
)
nabm_codes.loc[:, "sub_chapter_n1"] = nabm_codes.loc[:, "sub_chapter_n1"].map(
    lambda x: map_to_int(x)
)

# %%
path2nabm_hierarchy = "../resources/nabm_tree.csv"
nabm_codes.to_csv(path2nabm_hierarchy, index=False)

# %% [markdown]
# ### Convert nabm to loinc

# %%
# TODO only a mapping as json into ressources and assess its capability

# %%
kept_nabm_cols = {
    "concept_name": "nabm_concept_name",
    "concept_code": "nabm_concept_code",
    "concept_id_2": "loinc_concept_id",
}

kept_loinc_cols = {
    "concept_name": "loinc_concept_name",
    "concept_code": "loinc_concept_code",
    "concept_id": "loinc_concept_id",
}

snds_concepts = pd.read_csv(
    "../../../../omop_universe/drees_conceptual_mapping/snds_concepts_from_susana/CONCEPT.csv"
)
nabm_concepts = snds_concepts.loc[
    snds_concepts["vocabulary_id"] == "SNDS - nabm", :
]
snds_relations = pd.read_csv(
    "../../../../omop_universe/drees_conceptual_mapping/snds_concepts_from_susana/CONCEPT_RELATIONSHIP.csv"
)
athena_loinc_raw = pd.read_csv(
    "../../../../omop_universe/drees_conceptual_mapping/local_counts/athena_data/loinc_from_athena/CONCEPT.csv",
    sep="\t",
)
athena_loinc = athena_loinc_raw.loc[
    athena_loinc_raw["vocabulary_id"] == "LOINC", kept_loinc_cols.keys()
].rename(columns=kept_loinc_cols)

nabm_standard_concepts = (
    nabm_concepts.merge(
        snds_relations,
        left_on="concept_id",
        right_on="concept_id_1",
        how="left",
    )
    .loc[:, kept_nabm_cols.keys()]
    .rename(columns=kept_nabm_cols)
)

# %%
def map_to_int(x):
    if np.isnan(x):
        return 0
    else:
        return np.int64(x)


nabm_standard_concepts.loc[:, "loinc_concept_id"] = nabm_standard_concepts.loc[
    :, "loinc_concept_id"
].map(lambda x: map_to_int(x))

nabm2loinc_mapping = nabm_standard_concepts.merge(
    athena_loinc, left_on="loinc_concept_id", right_on="loinc_concept_id"
).drop("loinc_concept_id", axis=1)

# %%
nabm2loinc_mapping.to_csv("../resources/nabm2loinc.csv", index=False)

# %%
