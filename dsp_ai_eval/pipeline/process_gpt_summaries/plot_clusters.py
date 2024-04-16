import altair as alt
import ast
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sentence_transformers import SentenceTransformer

from nesta_ds_utils.viz.altair import saving as viz_save

from dsp_ai_eval import PROJECT_DIR, config
from dsp_ai_eval.getters.gpt import (
    get_gpt_themes_embeddings,
    get_representative_docs,
    get_topics,
    get_probs,
    get_topic_model,
    get_cluster_summaries_cleaned,
)
from dsp_ai_eval.utils.clustering_utils import create_df_for_viz

model = SentenceTransformer(config["embedding_model"])

SEED = config["seed"]


def convex_hull_points(df, group_label):
    points = df[["x", "y"]].to_numpy()
    hull = ConvexHull(points)
    hull_indices = np.append(hull.vertices, hull.vertices[0])
    hull_points = points[hull_indices]  # Get hull points in order
    hull_df = pd.DataFrame(hull_points, columns=["x", "y"])
    hull_df["gpt_model"] = group_label
    # Add an explicit 'order' column based on the DataFrame's index
    # This will help to maintain the order of points when plotting
    hull_df["order"] = range(len(hull_df))
    return hull_df


if __name__ == "__main__":
    answers_long = get_gpt_themes_embeddings()

    docs = answers_long["answer_cleaned"].tolist()
    answers_long["embeddings"] = answers_long["embeddings"].apply(ast.literal_eval)
    embeddings = answers_long["embeddings"].apply(pd.Series).values

    topic_model = get_topic_model()

    cluster_summaries = get_cluster_summaries_cleaned()

    topics = get_topics()
    probs = get_probs()
    representative_docs = get_representative_docs()

    df_vis = create_df_for_viz(embeddings, topic_model, topics, docs, seed=SEED)

    df_vis = df_vis.merge(cluster_summaries, on="topic", how="left")
    df_vis = df_vis.merge(
        answers_long[["answer_cleaned", "temperature", "gpt_model", "heading"]],
        left_index=True,
        right_index=True,
    )

    df_vis["topic_name"].fillna("NA", inplace=True)
    df_vis["heading"].fillna("NA", inplace=True)

    opacity_condition = alt.condition(
        alt.datum.topic_name == "NA", alt.value(0.2), alt.value(0.5)
    )

    fig = (
        alt.Chart(df_vis)
        .mark_circle(size=100)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            y=alt.Y("y:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            color=alt.Color("topic_name:N", legend=None).scale(
                scheme=config["gpt_themes_pipeline"]["cluster_colours"]
            ),
            opacity=opacity_condition,
            tooltip=["topic_name", "doc"],
        )
        .properties(width=800, height=600)
        .interactive()
    )

    fig.save(PROJECT_DIR / f"outputs/figures/gpt_clusters.html")
    viz_save.save(fig, f"gpt_clusters", PROJECT_DIR / "outputs/figures", save_png=True)

    # plot hull
    hull_df = pd.concat(
        [
            convex_hull_points(df_vis[df_vis["gpt_model"] == g], g)
            for g in df_vis["gpt_model"].unique()
        ],
        ignore_index=True,
    )

    opacity_condition = alt.condition(
        alt.datum.topic_name == "NA", alt.value(0.2), alt.value(0.5)
    )

    scatter_plot = (
        alt.Chart(df_vis)
        .mark_circle(size=100)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            y=alt.Y("y:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            color="gpt_model",
            opacity=opacity_condition,
            tooltip=["topic_name", "doc"],
        )
    )

    hull_plot = (
        alt.Chart(hull_df)
        .mark_line()
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            y=alt.Y("y:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
            color="gpt_model",
            order=alt.Order("order"),
        )
    )

    plot = (scatter_plot + hull_plot).properties(width=800, height=600).interactive()

    plot.save(PROJECT_DIR / f"outputs/figures/gpt_hull_plot.html")
    viz_save.save(
        plot, f"gpt_hull_plot", PROJECT_DIR / "outputs/figures", save_png=True
    )
