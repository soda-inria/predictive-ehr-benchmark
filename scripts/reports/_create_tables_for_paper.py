# %%
import pandas as pd
from medem.constants import MACE_CODES
from pathlib import Path

path2paper_img = Path(
    "/home/mdoutrel/projets/inria/papiers/event2vec_paper/img"
)
# %%
# mace codes
mace_code_clean = {k: ", ".join(v) for k, v in MACE_CODES.items()}
mace_code_clean["Other codes"] = mace_code_clean[
    "Other codes from top pathologies"
]
mace_code_clean.pop("Other codes from top pathologies")
mace_code_clean_pd = pd.DataFrame.from_dict(mace_code_clean, orient="index")
mace_code_clean_pd.to_latex(path2paper_img / "mace_codes.tex", header=False)
# %%
