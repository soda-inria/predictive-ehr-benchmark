# %%
import pandas as pd
import json
import re
import sys
sys.path.append('../src/')
from choiUtils import read_embedding_codes, read_cui_embedding_codes, get_matrix_fromcui2vec
from utils import get_local_as_cuis

%load_ext autoreload
%autoreload 2

# %%
path2cuis = '../resources/umls_subset.csv'
umls_cuis = pd.read_csv(path2cuis)

path2snds2vec = '../data/echantillon_mid_grain_r=90-centered2019-12-05_19:11:27/echantillon_mid_grain_r=90-centered2019-12-05_19:11:27.json'
with open(path2snds2vec, 'r') as f:
    snds2vecFG = json.load(f)

# %%
local2cuis = get_local_as_cuis(snds2vecFG, umls_cuis)
local2cuis.head()

# %% [markdown]
# # Mapping of snds2vec finegrain to cui

# %% [markdown]
# # choi vs snds2vec

# %%
#path2mcemc = '../data/claims_codes_hs_300.txt'
path2mcemc = '../data/claims_cuis_hs_300.txt'
embedding_matrix, embedding_type_to_indices, name_to_idx, idx_to_name = read_cui_embedding_codes(path2mcemc)

# %%
mcemc_and_snds2vec_common = set(name_to_idx.keys()) & set(local2cuis['CUI'].values)
print(len(mcemc_and_snds2vec_common))
local2cuis.loc[local2cuis['CUI'].isin(mcemc_and_snds2vec_common), 'SAB'].value_counts()

# %% [markdown]
# # cui2vec vs snds2vec

# %%
# loading only cui2vec embeddings matching with atc or icd10 from umls 
path2currated_cuis = '../data/cui2vec_act-icd10.csv'
cui2vec = pd.read_csv(path2currated_cuis)
cui2vec['SAB'].value_counts()

# %%
# common codes
cui2vec_and_snds2vec_common = set(cui2vec['CUI'].values) & set(local2cuis['CUI'].values)
print(len(cui2vec_and_snds2vec_common))
local2cuis.loc[local2cuis['CUI'].isin(cui2vec_and_snds2vec_common), 'SAB'].value_counts()

# %%



