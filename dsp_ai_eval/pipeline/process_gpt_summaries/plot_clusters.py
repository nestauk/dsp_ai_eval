import altair as alt
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull

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

SEED = config["seed"]


def convex_hull_points(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """
    Computes the convex hull of a set of points grouped by a specified label.
    (In practice, the points are grouped by GPT model.)

    Parameters:
        df (pd.DataFrame): DataFrame containing the points with columns 'x' and 'y'.
        group_label (str): Label used to identify the group in the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the convex hull points with additional columns for group label and order.
    """
    points = df[["x", "y"]].to_numpy()
    hull = ConvexHull(points)
    hull_indices = np.append(hull.vertices, hull.vertices[0])
    hull_points = points[hull_indices]  # Get hull points in order
    hull_df = pd.DataFrame(hull_points, columns=["x", "y"])
    hull_df["gpt_model"] = group_label
    # Add an explicit 'order' column based on the DataFrame's index
    # This will help to maintain the order of points when plotting so that the lines join up the points correctly
    hull_df["order"] = range(len(hull_df))
    return hull_df


def create_cluster_plot(
    df_vis: pd.DataFrame,
    point_size: int = 100,
    opacity_val: float = 0.25,
    color_scheme: str = config["gpt_themes_pipeline"]["cluster_colours"],
    html_outpath: Path = PROJECT_DIR / "outputs/figures/gpt_clusters.html",
    png_name: str = "gpt_clusters",
    png_outpath: Path = PROJECT_DIR / "outputs/figures",
) -> None:
    """
    Generates a scatter plot with jittered x- and y-coordinates and saves it.

    Parameters:
        df_vis (pd.DataFrame): DataFrame containing visualization data.
        point_size (int, optional): Size of points in the plot. Defaults to 100.
        opacity_val (float, optional): Opacity value of the points. Defaults to 0.25.
        color_scheme (str, optional): Color scheme identifier for plotting.
        html_outpath (Path, optional): Output path for saving the HTML version of the plot. Defaults to configured path.
        png_name (str, optional): Base name for the PNG file. Defaults to 'gpt_clusters'.
        png_outpath (Path, optional): Output path for saving the PNG file. Defaults to configured path.

    Returns:
        None: Saves the generated plot to specified file paths.
    """

    fig = (
        alt.Chart(df_vis)
        .mark_circle(size=point_size)
        .transform_calculate(
            jittered_x="datum.x + sqrt(-2*log(random()))*cos(2*PI*random())*0.5",
            jittered_y="datum.y + sqrt(-2*log(random()))*sin(2*PI*random())*0.5",
        )
        .encode(
            x=alt.X(
                "jittered_x:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            y=alt.Y(
                "jittered_y:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            color=alt.Color("topic_name:N", legend=None).scale(scheme=color_scheme),
            opacity=alt.value(opacity_val),
            tooltip=["topic_name:N", "doc:N"],
        )
        .properties(width=900, height=600)
        .interactive()
    )

    fig.save(html_outpath)
    viz_save.save(fig, png_name, png_outpath, save_png=True)


def create_hull_plot(
    df_vis: pd.DataFrame,
    point_size: int = 100,
    html_outpath: Path = PROJECT_DIR / "outputs/figures/gpt_hull_plot.html",
    png_name: str = "gpt_hull_plot",
    png_outpath: Path = PROJECT_DIR / "outputs/figures",
) -> None:
    """
    Plots the summaries produced by different GPT models and plots their convex hulls,
    with the purpose of seeing to what extent the areas covered by each model overlap.

    Parameters:
        df_vis (pd.DataFrame): DataFrame containing visualization data.
        point_size (int, optional): Size of the points in the plot. Defaults to 100.
        html_outpath (Path, optional): Output path for the HTML version of the plot. Defaults to a configured path.
        png_name (str, optional): Base name for the PNG file. Defaults to 'gpt_hull_plot'.
        png_outpath (Path, optional): Output path for saving the PNG file. Defaults to a configured path.

    Returns:
        None: Saves the combined scatter and hull plot to specified file paths.
    """
    hull_df = pd.concat(
        [
            convex_hull_points(df_vis[df_vis["gpt_model"] == g], g)
            for g in df_vis["gpt_model"].unique()
        ],
        ignore_index=True,
    )

    opacity_condition = alt.condition(
        alt.datum.topic_name == "NA", alt.value(0.1), alt.value(0.4)
    )

    scatter_plot = (
        alt.Chart(df_vis)
        .mark_circle(size=point_size)
        .encode(
            x=alt.X(
                "x:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
            ),
            y=alt.Y(
                "y:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
            ),
            color="gpt_model",
            opacity=opacity_condition,
            tooltip=["topic_name", "doc"],
        )
    )

    hull_plot = (
        alt.Chart(hull_df)
        .mark_line()
        .encode(
            x=alt.X(
                "x:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
            ),
            y=alt.Y(
                "y:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
            ),
            color="gpt_model",
            order=alt.Order("order"),
        )
    )

    plot = (scatter_plot + hull_plot).properties(width=800, height=600).interactive()

    plot.save(html_outpath)
    viz_save.save(plot, png_name, png_outpath, save_png=True)


if __name__ == "__main__":
    answers_long = get_gpt_themes_embeddings()

    docs = answers_long["answer_cleaned"].tolist()
    answers_long["embeddings"] = answers_long["embeddings"].apply(ast.literal_eval)
    embeddings = answers_long["embeddings"].apply(pd.Series).values

    topic_model = get_topic_model()

    cluster_summaries = get_cluster_summaries_cleaned()

    topics = get_topics()
    probs = get_probs()
    # representative_docs = get_representative_docs() # Not used

    df_vis = create_df_for_viz(embeddings, topic_model, topics, docs, seed=SEED)

    df_vis = df_vis.merge(cluster_summaries, on="topic", how="left")
    df_vis = df_vis.merge(
        answers_long[["answer_cleaned", "temperature", "gpt_model", "heading"]],
        left_index=True,
        right_index=True,
    )

    df_vis["topic_name"].fillna("NA", inplace=True)
    df_vis["heading"].fillna("NA", inplace=True)

    create_cluster_plot(df_vis)

    create_hull_plot(df_vis)
