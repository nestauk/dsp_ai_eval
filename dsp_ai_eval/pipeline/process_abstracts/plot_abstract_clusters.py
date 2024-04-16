import altair as alt
from bertopic import BERTopic
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from nesta_ds_utils.viz.altair import saving as viz_save

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils.clustering_utils import create_df_for_viz
from dsp_ai_eval.getters.scite import (
    get_scite_df_w_embeddings,
    get_topic_model,
    get_probs,
    get_representative_docs,
    get_topics,
    get_cluster_summaries_clean,
)

# Increase the maximum number of rows Altair will process
alt.data_transformers.disable_max_rows()

embedding_model = SentenceTransformer(config["embedding_model"])

SEED = config["seed"]


def create_chart(df_vis, scale_by_citations=False, save=True, filename_suffix=""):
    # Conditional field calculation based on `scale_by_citations` flag
    if scale_by_citations:
        size_encode = alt.Size(
            "size_calculated:Q",
            scale=alt.Scale(range=[0, 2000]),
            legend=alt.Legend(title="N citations"),
        )

        tooltip_fields = ["topic_name:N", "doc:N", "total_cites:N"]
    else:
        size_encode = alt.value(50)

        tooltip_fields = ["topic_name:N", "doc:N"]

    opacity_condition = alt.condition(
        alt.datum.topic_name == "NA", alt.value(0.2), alt.value(0.5)
    )

    # Base chart setup
    if scale_by_citations:
        chart = alt.Chart(df_vis).transform_calculate(
            # Create a new field for size, multiplying 'total_cites' by 10
            size_calculated="datum.total_cites * 10"  #'log(datum.total_cites + 1)' # Tried log-scaling but it didn't look great
        )
    else:
        chart = alt.Chart(df_vis)

    base = chart.encode(
        x=alt.X("x:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
        y=alt.Y("y:Q", axis=alt.Axis(ticks=False, labels=False, title=None)),
        size=size_encode,
        color=alt.Color("topic_name:N", legend=None).scale(
            scheme=config["abstracts_pipeline"]["cluster_colours"]
        ),
        opacity=opacity_condition,
        tooltip=tooltip_fields,
    ).mark_circle()

    plot = (base).properties(width=800, height=600).interactive()

    # Saving logic, customized based on parameters
    if save:
        filename = f"scite_abstracts{filename_suffix}.html"
        plot.save(PROJECT_DIR / f"outputs/figures/{filename}")
        viz_save.save(
            plot,
            f"scite_abstracts{filename_suffix}",
            PROJECT_DIR / "outputs/figures",
            save_png=True,
        )


if __name__ == "__main__":
    scite_abstracts = get_scite_df_w_embeddings()

    docs = scite_abstracts["title_abstract"].to_list()
    embeddings = scite_abstracts["embeddings"].apply(pd.Series).values

    topic_model = get_topic_model()

    cluster_summaries = get_cluster_summaries_clean()

    topics = get_topics()
    probs = get_probs()
    representative_docs = get_representative_docs()

    df_vis = create_df_for_viz(embeddings, topic_model, topics, docs, seed=SEED)

    df_vis = df_vis.merge(
        scite_abstracts, left_on="doc", right_on="title_abstract", how="left"
    )

    df_vis = df_vis.merge(
        cluster_summaries, left_on="topic", right_on="topic", how="left"
    )

    df_vis["topic_name"].fillna("NA", inplace=True)

    create_chart(df_vis, scale_by_citations=False, filename_suffix="")

    create_chart(df_vis, scale_by_citations=True, filename_suffix="_citations")
