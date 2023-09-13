# %%
from medem.constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# %%

behrt_performances = pd.read_csv(
    DIR2RESOURCES / "behrt-disease-performances.csv"
)
x_label = "Number of cases"
N = 391000
behrt_performances[x_label] = behrt_performances["Ratio"] * N
# %%
# Is there a ratio trend ?

fig, axes = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
sns.scatterplot(
    data=behrt_performances,
    x=x_label,
    y="AUROC",
    ax=axes[0],
)
sns.scatterplot(
    data=behrt_performances,
    x=x_label,
    y="APS",
    ax=axes[1],
    color="orange",
)
axes[0].set_xlim(0, 0.08 * N)
axes[1].set_ylim(0, 0.8)
plt.savefig(DIR2DOCS_IMG / "behrt-performances.png", bbox_inches="tight")
plt.savefig(DIR2DOCS_IMG / "behrt-performances.pdf", bbox_inches="tight")
plt.plot()
# %%
### explore with plotly
p = px.scatter(
    data_frame=behrt_performances,
    x="Ratio",
    y="AUROC",
    hover_data="Description",
)
p
# %%
px.scatter(
    data_frame=behrt_performances, x="Ratio", y="APS", hover_data="Description"
)

# %%
