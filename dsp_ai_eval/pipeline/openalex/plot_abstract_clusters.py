import altair as alt
import pandas as pd

from nesta_ds_utils.viz.altair import saving as viz_save

from dsp_ai_eval import PROJECT_DIR, S3_BUCKET, logger
from dsp_ai_eval.utils.clustering_utils import create_df_for_viz
from dsp_ai_eval.getters.utils import save_to_s3
from dsp_ai_eval.getters.openalex import (
    get_openalex_df_w_embeddings,
    get_topic_model,
    get_topics,
    get_cluster_summaries_clean,
)

import typer
from typing import Any, Dict, Optional
from typing_extensions import Annotated
from pathlib import Path

# Increase the maximum number of rows Altair will process
alt.data_transformers.disable_max_rows()


def create_themes_barchart(
    pipeline: str,
    save: bool = True,
    filename_suffix: str = "",
) -> None:
    topics_counts = (
        get_topic_model(pipeline=pipeline)
        .get_topic_info()[["Topic", "Count"]]
        .query("Topic != -1")
    )
    topic_names = get_cluster_summaries_clean(pipeline=pipeline)[
        ["topic_name", "topic"]
    ]
    barchart_df = topics_counts.merge(topic_names, left_on="Topic", right_on="topic")

    hbarchart = (
        alt.Chart(barchart_df)
        .encode(
            x=alt.X("Count:Q", title="Number of documents"),
            y=alt.Y("topic_name:N", title="Themes"),
            # TODO: modify topic_name text to be more readable (large font)
            tooltip=["topic_name:N", "Count:Q"],
        )
        .mark_bar()
    )

    plot = hbarchart.properties(width=800, height=600).interactive()

    # Save the chart
    if save:
        filename = f"{pipeline}_hbarchart_{filename_suffix}.html"
        plot.save(str(PROJECT_DIR / f"outputs/figures/{filename}"))
        viz_save.save(
            plot,
            f"{pipeline}_hbarchart_{filename_suffix}",
            PROJECT_DIR / "outputs/figures",
            save_png=True,
        )


def map_citations_to_size(citations: int, quantile_values: dict) -> str:
    """
    Maps the number of citations to a size category based on quantile thresholds.
    The string output can be used to set point sizes in an Altair chart.

    Parameters:
        citations (int): The number of citations.
        quantile_values (dict): A dictionary with quantiles as keys and the corresponding threshold values as values.

    Returns:
        str: A string indicating the citation size category.
    """

    a = int(quantile_values[0.25])
    b = int(quantile_values[0.5])
    c = int(quantile_values[0.75])

    if citations < a:
        return f"5-{a-1}"
    elif a <= citations < b:
        return f"{a}-{b-1}"
    elif b <= citations < c:
        return f"{b}-{c-1}"
    else:
        return f"{c}+"


def create_chart(
    df_vis: pd.DataFrame,
    scale_by_citations: bool = False,
    save: bool = True,
    filename_suffix: str = "",
    add_topic_legend: bool = False,
    plot_na=False,
    cluster_colours: str = "tableau20",
) -> None:
    """
    Generates and optionally saves a visualization chart based on citation data.

    Parameters:
        df_vis (pd.DataFrame): A DataFrame containing visualization data, must include a column 'total_cites'.
        scale_by_citations (bool, optional): A flag to determine if the size of the points in the chart should be scaled
            based on the number of citations. Defaults to False.
        save (bool, optional): A flag to determine if the chart should be saved to a file. Defaults to True.
        filename_suffix (str, optional): A suffix to append to the filename when saving the chart. Defaults to an empty string.

    Returns:
        None: The function generates and displays a chart, and optionally saves it.
    """
    if not plot_na:
        df_vis = df_vis.query("topic != -1")

    # Create quantile bins for scaling the point size in the chart
    quantile_values = df_vis["total_cites"].quantile([0.25, 0.5, 0.75])
    logger.info(f"quantiles: {quantile_values}")

    df_vis["point_size"] = df_vis["total_cites"].apply(
        lambda x: map_citations_to_size(x, quantile_values)
    )

    logger.info(df_vis["point_size"].value_counts(dropna=False))

    # Conditional field calculation based on `scale_by_citations` flag
    if scale_by_citations:

        a = int(quantile_values[0.25])
        b = int(quantile_values[0.5])
        c = int(quantile_values[0.75])

        # Create a calculated field for binning citation counts into categories
        chart = alt.Chart(df_vis)
        size_encode = alt.Size(
            "point_size:O",  # 'O' for ordinal since we're using discrete categories
            scale=alt.Scale(
                domain=[
                    f"5-{a-1}",
                    f"{a}-{b-1}",
                    f"{b}-{c-1}",
                    f"{c}+",
                ],  # These should match the outputs from map_citations_to_size
                range=[50, 200, 500, 1000],
            ),
            legend=alt.Legend(
                title="Number of citations",
                titleFontSize=12,
                labelPadding=100,
                labelFontSize=12,
            ),
        )
        tooltip_fields = ["topic_name:N", "doc:N", "total_cites:N", "point_size:O"]
    else:
        chart = alt.Chart(df_vis)
        size_encode = alt.value(50)  # Use a constant size when not scaling by citations
        tooltip_fields = ["topic_name:N", "doc:N"]

    # Make the noise cluster more transparent than the other clusters
    opacity_condition = alt.condition(
        alt.datum.topic_name == "NA", alt.value(0.25), alt.value(0.6)
    )

    if add_topic_legend:
        color_args = (
            alt.Color("topic_name:N")
            .legend(orient="top-left", labelLimit=0)
            .title("Topics by Colour")
            .scale(scheme=cluster_colours)
        )
    else:
        color_args = (
            alt.Color("topic_name:N").legend(None).scale(scheme=cluster_colours)
        )

    base = chart.encode(
        x=alt.X(
            "x:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
        ),
        y=alt.Y(
            "y:Q", axis=alt.Axis(ticks=False, labels=False, title=None, grid=False)
        ),
        size=size_encode,
        color=color_args,
        opacity=opacity_condition,
        tooltip=tooltip_fields,
    ).mark_circle()

    plot = (base).properties(width=800, height=600).interactive()

    # Save the chart
    if save:
        filename = f"openalex_abstracts{filename_suffix}.html"
        plot.save(str(PROJECT_DIR / f"outputs/figures/{filename}"), format="html")
        viz_save.save(
            plot,
            f"openalex_abstracts{filename_suffix}",
            PROJECT_DIR / "outputs/figures",
            save_png=True,
        )


def run_pipeline(
    rq_prefix: str,
    path_vis_data: str,
    seed: int,
):
    pipeline_name = "openalex_abstracts"
    df = get_openalex_df_w_embeddings()

    if "cited_by_count" in df.columns:
        df = df.rename(columns={"cited_by_count": "total_cites"})

    docs = df["title_abstract"].to_list()
    embeddings = df["embeddings"].apply(pd.Series).values

    topic_model = get_topic_model(pipeline=pipeline_name)

    cluster_summaries = get_cluster_summaries_clean(pipeline=pipeline_name).astype(
        {"topic": int}
    )

    topics = get_topics(pipeline=pipeline_name)

    df_vis = create_df_for_viz(embeddings, topic_model, topics, docs, seed=seed).astype(
        {"topic": int}
    )

    df_vis = df_vis.merge(df, left_on="doc", right_on="title_abstract", how="left")

    df_vis = df_vis.merge(
        cluster_summaries, left_on="topic", right_on="topic", how="left"
    )

    df_vis["topic_name"].fillna("NA", inplace=True)

    # Drop nested columns
    df_vis = df_vis.drop(
        columns=[
            "ids",
            "apc_list",
            "apc_paid",
            "cited_by_percentile_year",
            "biblio",
            "primary_topic",
            "topics",
            "keywords",
            "concepts",
            "mesh",
            "locations",
            "best_oa_location",
            "primary_location",
            "open_access",
            "authorships",
            "sustainable_development_goals",
            "grants",
            "datasets",
            "versions",
            "counts_by_year",
        ]
    )

    save_to_s3(
        S3_BUCKET,
        df_vis,
        f"{rq_prefix}/{path_vis_data}",
    )

    df_vis = df_vis[
        ["topic", "topic_name", "total_cites", "doc", "x", "y"]
    ]  # Select specific cols to avoid JSON error

    create_chart(
        df_vis, scale_by_citations=False, filename_suffix="", add_topic_legend=True
    )

    create_chart(
        df_vis,
        scale_by_citations=True,
        filename_suffix="_citations",
        add_topic_legend=True,
    )


if __name__ == "__main__":
    from dsp_ai_eval import config

    run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_vis_data=config["oa_abstracts_pipeline"]["path_vis_data"],
        seed=config["seed"],
    )
