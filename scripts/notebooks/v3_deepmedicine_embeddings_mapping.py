# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from medem.constants import *
from medem.experiences.configurations import PATH2CUI2VEC_FIRST_EMBEDDINGS, PATH2CUI2VEC_MEAN_EMBEDDINGS
import re
# %%
cbow_embeddings = pd.read_parquet(DIR2EMBEDDINGS / "medical-concept-embeddings" / "diseases_emb_CBOW.parquet")
# %%
cbow_embeddings