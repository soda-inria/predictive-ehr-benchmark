# %%
"""
Visualize the embeddings of APHP with TSNE plot.
"""

# To visualize in two dimension the embeddings, we can use t-SNE, a parametric
# gaussian dimension reduction favoring the apparition of groups.
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import json
from event2vec.config import DIR2RESULTS

# %%
dir2snds_embeddings = DIR2RESULTS / "omop_sample_4tables_90d"

concepts_labels = pd.read_csv(dir2snds_embeddings / "concept_labels.csv")
concepts_labels["vocabulary_id"] = (
    concepts_labels["vocabulary_id"]
    .str.replace("APHP - ITM - ", "")
    .str.replace("APHP - ORBIS - ", "")
    .str.replace("Medicament - ATC & Article", "ATC")
    .str.upper()
)
concepts_labels["concept_id"] = concepts_labels["concept_id"].astype(str)
with open(
    dir2snds_embeddings / "tuto_snds2vec_alpha=0.75_k=1_d=150.json",
    "r",
) as f:
    embeddings_dict = json.load(f)
embeddings_matrix = np.matrix(list(embeddings_dict.values()))
# %%
vocabulary_names = np.sort(concepts_labels["vocabulary_id"].unique())
color_list = [
    "orchid",
    "limegreen",
    "firebrick",
    "deepskyblue",
]
color_map = dict(zip(vocabulary_names, color_list[: len(vocabulary_names)]))
concepts_labels["plotted_label"] = concepts_labels.apply(
    lambda x: str(x["concept_code"]) + " : " + str(x["concept_name"]), axis=1
)  # ['concept_code']= .to_dict()
concepts_labels["plotted_color"] = (
    concepts_labels["vocabulary_id"].map(lambda x: color_map[x]).to_list()
)
# %%
# ## Fit the TSNE
perplexity = 30
n_iter = 1000
metric = "cosine"
rs = 2
tsne = TSNE(
    n_components=2,
    metric=metric,
    perplexity=perplexity,
    early_exaggeration=15,
    n_iter=n_iter,
    random_state=rs,
    n_jobs=-1,
)
X = tsne.fit_transform(embeddings_matrix)
# %%
# ## Interactive plot

coordinates = pd.DataFrame(
    {"concept_id": list(embeddings_dict.keys()), "x": X[:, 0], "y": X[:, 1]}
)
plotted_concepts = concepts_labels.merge(
    coordinates, on="concept_id", how="inner"
)
print(plotted_concepts.shape)
print(concepts_labels.shape)

import plotly
import plotly.graph_objects as go

fig = go.Figure(
    layout=dict(
        # title="{}".format('snds2vec'),
        autosize=True,
        width=1000,
        height=1000,
        xaxis={"title": "x", "showticklabels": True},
        yaxis={"title": "y", "showticklabels": False},
        margin={"l": 0, "b": 0, "t": 50, "r": 0},
        showlegend=True,
        hovermode="closest",
    )
)

for vocab in vocabulary_names:
    vocab_data = plotted_concepts.loc[
        plotted_concepts["vocabulary_id"] == vocab, :
    ]
    fig.add_trace(
        go.Scattergl(
            x=vocab_data["x"],
            y=vocab_data["y"],
            legendgroup="Vocabulary",
            name=vocab,
            text=vocab_data["plotted_label"],
            mode="markers",
            opacity=0.8,
            marker={"size": 6, "color": vocab_data["plotted_color"]},
        )
    )

fig.update_layout(
    legend=dict(
        x=0, y=1, font=dict(size=25), title="Vocabulary", itemsizing="constant"
    )
)
config = {"scrollZoom": True}

fig.show(renderer="jupyterlab", config=config)
plotly.offline.plot(
    fig,
    filename=str(
        dir2snds_embeddings
        / f"tsne_plot__metric={metric}_perplexity={perplexity}_niter={n_iter}_rs={rs}.html"
    ),
    config=config,
)
# -
