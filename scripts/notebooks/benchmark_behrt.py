# %%
import pandas as pd
from medem.constants import DIR2RESOURCES

# %%
behrt_results = pd.read_csv(
    DIR2RESOURCES / "benchmarks" / "behrt_next_6_months_caliber.csv",
)
caliber_chapter = "Diseases of the circulatory system"
behrt_results.loc[behrt_results["Caliber Chapter"] == caliber_chapter][
    ["APS", "AUROC", "Ratio"]
].describe()
# %%
