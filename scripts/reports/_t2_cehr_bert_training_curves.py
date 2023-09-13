# %%
import pandas as pd 
import numpy as np
from medem.experiences.configurations import CONFIG_PROGNOSIS_COHORT, cohort_configuration_to_str
from medem.experiences.utils import ICD10_LABEL2CHAPTER
from medem.reports.utils import collect_cehr_bert_logs_prognosis

import seaborn as sns
import matplotlib.pyplot as plt

from tableone import TableOne
from medem.constants import *
from pathlib import Path
from copy import deepcopy
pd.set_option("display.max_columns", 50)


# %%
def load_history(dir2training_history: Path):
    """Add random seed to history if not existing. 
    
    """
    history = pd.read_parquet(dir2training_history)
    if "seed" not in history.columns:
        seed = 0
        loss = np.inf
        line_count = 0
        line_start = 0
        history["seed"] = -1
        history["epoch"] = -1
        for ll in list(history.iterrows())[1:]:
            line_count+=1
            if np.round(ll[1]["lr"], 6) == 0.0001:
                history.iloc[line_start:line_start+line_count, -2] = seed
                history.iloc[line_start:line_start+line_count, -1] = np.arange(0, line_count)
                seed+=1
                line_start=line_start+line_count
                line_count=0
    return history


# %%
config = deepcopy(CONFIG_PROGNOSIS_COHORT)
config.pop("target_chapter", None)

model_name="CEHR_BERT_512_hospital_split"

dir2cohort = DIR2DATA / cohort_configuration_to_str(config)
dir2evaluation = dir2cohort/"evaluation_train_val_split"

target = 11
dir2training_history_split = dir2evaluation / f"{model_name}__target_{target}"/"history"
history_split = load_history(dir2training_history_split)
history_split = history_split.loc[(history_split["seed"]<5)&(history_split["seed"]!=-1)]
history_wo_hospital_split = load_history(dir2evaluation / f"CEHR_BERT_512_pr2_pipeline__target_{target}"/"history")
history_wo_hospital_split = history_wo_hospital_split.loc[(history_wo_hospital_split["seed"]<5)&(history_wo_hospital_split["seed"]!=-1)]


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
for ax_id, (train_history_, train_history_name) in enumerate(zip([history_split, history_wo_hospital_split], ["Random val split by hospital id", "Fixed validation split without hospital id"])):
    data_to_plot = train_history_.melt(id_vars=["seed", "epoch"], value_vars=["val_loss", "loss"], var_name="dataset", value_name="epoch_loss")
    sns.lineplot(
        data=data_to_plot,
        x="epoch",
        y="epoch_loss",
        hue="dataset",
        ax=axes[ax_id],
        units="seed",
        estimator=None
    )
    sns.lineplot(
        data=data_to_plot,
        x="epoch",
        y="epoch_loss",
        hue="dataset",
        ax=axes[ax_id],
        lw=0,
    )
    axes[ax_id].set_title(train_history_name)

# %%
DIR2DOCS/
