# %%
from medem.constants import *
import pandas as pd
from medem.experiences.configurations import PATH2SNDS_EMBEDDINGS
import numpy as np

pd.set_option("display.max_colwidth", 1000)
# %%
nabm2loinc_2020 = pd.read_csv(
    DIR2RESOURCES / "nabm2loinc_2020.csv", dtype={"nabm_concept_code": str}
)
nabm2loinc_2023 = (
    pd.read_excel(
        DIR2RESOURCES / "LOINCFR_JeuDeValeurs_2023.xlsx",
        sheet_name="Alignement NABM-LOINC FR",
    )
    .rename(
        columns={
            "Code NABM": "nabm_concept_code",
            "Libellé NABM": "nabm_concept_name",
            "Code LOINC": "loinc_concept_code",
            "Libelle LOINC": "loinc_concept_name",
        }
    )
    .astype({"nabm_concept_code": str, "loinc_concept_code": str})
)
print("nabm2loinc 2020: ", nabm2loinc_2020.shape)
print("nabm2loinc 2023: ", nabm2loinc_2023.shape)
nabm2loinc_2020.head()
snds_embeddings = pd.read_parquet(PATH2SNDS_EMBEDDINGS)
snds_vocabulary = pd.read_csv(
    PATH2SNDS_EMBEDDINGS.parent / "concept_labels.csv"
)
# %%
nabm_combined = pd.concat([nabm2loinc_2020, nabm2loinc_2023]).drop_duplicates()
# %%
for nabm2loinc_vocab, mapping_year in zip(
    [nabm2loinc_2020, nabm2loinc_2023], [2020, 2023]
):
    snds_nabm_subvocabulary = snds_vocabulary.loc[
        snds_vocabulary["vocabulary_id"] == "nabm", "concept_code"
    ].astype(str)
    nb_snds_nabm_vocab = len(snds_nabm_subvocabulary)
    existing_nabm_vocab = nabm2loinc_vocab["nabm_concept_code"].unique()
    nb_existing_nabm_vocab = len(existing_nabm_vocab)

    intersection_nabm = set(existing_nabm_vocab).intersection(
        set(snds_nabm_subvocabulary)
    )
    differences_nabm = set(snds_nabm_subvocabulary).difference(
        set(existing_nabm_vocab)
    )
    nb_intersection_nabm = len(intersection_nabm)

    print(nb_snds_nabm_vocab, nb_existing_nabm_vocab)

    print(
        f"Intersection between snds nabm codes and loinc mapping in {mapping_year}: {nb_intersection_nabm / nb_snds_nabm_vocab:.2f}={nb_intersection_nabm}/{nb_snds_nabm_vocab}"
    )
# %% [markdown]
# A large part of nabm code in the snds vocabulary has a loinc mapping.
# The mapping should be done from loinc to NABM since loinc is finer than NABM.

# %%
print(nabm2loinc_2020.shape)
nabm2loinc_2020[["nabm_concept_code", "loinc_concept_code"]].drop_duplicates()
# So there are multiple nabm codes for the same loinc, but looking at the
# labels, they are often closely related, meaning there is some redundancy in
# the NABM. %%
nabm2loinc_2020.drop_duplicates()

# %%
# and for 2023 ?
print(nabm2loinc_2023.shape)
nabm2loinc_2023[["nabm_concept_code", "loinc_concept_code"]].drop_duplicates()
# So there are multiple nabm codes for the same loinc
nabm2loinc_2023.sort_values("loinc_concept_code")[:100]

# %%

# I create an arbitraty mapping choice
loinc2nabm_mapper = (
    nabm2loinc_2020.groupby("loinc_concept_code")
    .agg(
        **{
            COLNAME_TARGET_CODE: pd.NamedAgg("nabm_concept_code", "first"),
            "nb_nabm_concept_codes": pd.NamedAgg(
                "nabm_concept_code", lambda x: len(np.unique(x))
            ),
        }
    )
    .reset_index()
    .rename(columns={"loinc_concept_code": COLNAME_SOURCE_CODE})
)
loinc2nabm_mapper.drop("nb_nabm_concept_codes", axis=1).to_csv(
    DIR2RESOURCES / "mapping_loinc2nabm_2020.csv", index=False
)
# %%
loinc2nabm_mapper.loc[loinc2nabm_mapper["nb_nabm_concept_codes"] > 1]
loinc2nabm_mapper["nabm_concept_code"].nunique()
