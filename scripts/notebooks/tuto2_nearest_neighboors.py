# %%
"""
Get KNN for some concepts (eg. E10) for APHP vs SNDS embeddings
"""

import numpy as np
import pandas as pd
import json
from event2vec.config import DIR2RESULTS
from event2vec.concept_proximity import get_closest_nn

pd.set_option("display.max_colwidth", 1000)

# paper_dir = "home/matthieu/boulot/these/papiers/event2vec"
# %%
dir2snds_embeddings = DIR2RESULTS / "snds"

concept_labels = pd.read_csv(dir2snds_embeddings / "concept_labels.csv")
if "concept_id" not in concept_labels.columns:
    concept_labels["concept_id"] = concept_labels["concept_code"]
concept_labels["concept_name"] = concept_labels[
    "concept_name"
].str.capitalize()
with open(
    dir2snds_embeddings
    / "echantillon_mid_grain_r=90-centered2019-12-05_19:11:27.json",
    "r",
) as f:
    snds_embeddings = json.load(f)
# %%
k = 10

source_concept_code = "E101"
# For SNDS
top_k_concepts = get_closest_nn(
    source_concept_code=source_concept_code,
    embedding_dict=snds_embeddings,
    concept_labels=concept_labels,
    k=10,
)
# Pretty printing
top_k_concepts["similarity"] = top_k_concepts["similarity"].round(2)
top_k_concepts["concept_code"] = (
    top_k_concepts["vocabulary_id"] + ":" + top_k_concepts["concept_code"]
)
print(
    top_k_concepts[["concept_code", "concept_name", "similarity"]].to_latex(
        index=False
    )
)
# %%
# Aphp
# %%

dir2aphp_embeddings = DIR2RESULTS / "omop_sample_4tables_90d"

aphp_concept_labels = pd.read_csv(dir2aphp_embeddings / "concept_labels.csv")
aphp_concept_labels["concept_id"] = aphp_concept_labels["concept_id"].astype(
    str
)
with open(
    dir2aphp_embeddings / "omop_200K_sample_4tables_alpha=0.75_k=1_d=150.json",
    "r",
) as f:
    aphp_embeddings = json.load(f)

# %%
k = 10

source_concept_code = "E101"
# For APHP
top_k_concepts = get_closest_nn(
    source_concept_code=source_concept_code,
    embedding_dict=aphp_embeddings,
    concept_labels=aphp_concept_labels,
    k=k,
)
# pretty printing
top_k_concepts["vocabulary_id"] = top_k_concepts["vocabulary_id"].str.replace(
    "APHP - ORBIS - ", ""
)
top_k_concepts["similarity"] = top_k_concepts["similarity"].round(2)
top_k_concepts["concept_code"] = (
    top_k_concepts["vocabulary_id"] + ":" + top_k_concepts["concept_code"]
)
print(
    top_k_concepts[["concept_code", "concept_name", "similarity"]].to_latex(
        index=False
    )
)

# %%
top_k_concepts.loc[top_k_concepts["vocabulary_id"] != "APHP - ORBIS - cim10"]
