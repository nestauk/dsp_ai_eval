import altair as alt
from bertopic import BERTopic
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from nesta_ds_utils.viz.altair import saving as viz_save

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils.clustering_utils import create_df_for_viz

# Increase the maximum number of rows Altair will process
alt.data_transformers.disable_max_rows()

embedding_model = SentenceTransformer(config["embedding_model"])

ABSTRACTS_EMBEDDINGS_INPATH = (
    PROJECT_DIR / "inputs/data/embeddings/scite_embeddings.parquet"
)
TOPICS_INPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_topics.pkl"
PROBS_INPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_probs.npy"
REPRESENTATIVE_DOCS_INPATH = (
    PROJECT_DIR / "outputs/data/bertopic_abstracts_representative_docs.pkl"
)
TOPIC_MODEL_INPATH = PROJECT_DIR / "outputs/models/bertopic_abstracts_model"
CLUSTER_SUMMARIES_INPATH = PROJECT_DIR / "outputs/data/cluster_summaries_cleaned.csv"

SEED = 42


def basic_chart(df_vis, save=True):
    base = alt.Chart(df_vis).encode(
        x="x",
        y="y",
        size=alt.condition(
            alt.datum.category == "main",  # Condition for the 'category' column
            alt.value(200),  # If True, size is 50
            alt.value(30),  # If False, size is 30
        ),
        opacity=alt.condition(
            alt.datum.category == "main",  # Condition for the 'topic' column
            alt.value(1),
            alt.value(0.5),  # If False, opacity is 0.5
        ),
        tooltip=["topic_name:N", "doc:N"],
    )

    # Chart for 'main' category points
    main_points = (
        base.transform_filter(alt.datum.category == "main")
        .mark_circle()
        .encode(color=alt.value("red"))  # Color is red for 'main'
    )

    # Chart for other points, colored by 'Name'
    other_points = (
        base.transform_filter(alt.datum.category != "main")
        .mark_circle()
        .encode(color="topic_name:N")  # Color mapped by 'Name'
    )

    # Combine the charts
    plot = (
        (main_points + other_points)
        .properties(
            width=800,
            height=600,
        )
        .interactive()
    )

    if save == True:
        plot.save(PROJECT_DIR / "outputs/figures/scite_abstracts.html")
        viz_save.save(
            plot, "scite_abstracts", PROJECT_DIR / "outputs/figures", save_png=True
        )

    plot.display()


def chart_scaled_by_citations(df_vis, save=True):
    # Define the base chart with common encodings
    base = (
        alt.Chart(df_vis)
        .transform_calculate(
            # Create a new field for size, multiplying 'total_cites' by 10
            size_calculated="datum.total_cites * 10"  #'log(datum.total_cites + 1)' # Tried log-scaling but it didn't look great
        )
        .encode(
            x="x",
            y="y",
            size=alt.Size(
                "size_calculated:Q",
                scale=alt.Scale(range=[0, 2000]),
                legend=alt.Legend(title="N citations"),
            ),  # Use the calculated field for size
            opacity=alt.condition(
                alt.datum.topic == "-1", alt.value(0.25), alt.value(0.5)
            ),
            tooltip=["topic_name:N", "doc:N", "total_cites:N"],
        )
    )

    # Chart for 'main' category points
    main_points = (
        base.transform_filter(alt.datum.category == "main")
        .mark_circle()
        .encode(color=alt.value("red"))  # Color is red for 'main'
    )

    # Chart for other points, colored by 'Name'
    other_points = (
        base.transform_filter(alt.datum.category != "main")
        .mark_circle()
        .encode(
            color=alt.Color("topic_name:N", legend=None),
        )
    )

    # Combine the charts
    plot = (
        (main_points + other_points)
        .properties(
            width=800,
            height=600,
        )
        .interactive()
    )

    if save == True:
        plot.save(PROJECT_DIR / "outputs/figures/scite_abstracts_citations.html")
        viz_save.save(
            plot,
            "scite_abstracts_citations",
            PROJECT_DIR / "outputs/figures",
            save_png=True,
        )


if __name__ == "__main__":
    scite_abstracts = pd.read_parquet(ABSTRACTS_EMBEDDINGS_INPATH)

    docs = scite_abstracts["title_abstract"].to_list()
    embeddings = scite_abstracts["embeddings"].apply(pd.Series).values

    topic_model = BERTopic.load(TOPIC_MODEL_INPATH, embedding_model=embedding_model)

    cluster_summaries = pd.read_csv(CLUSTER_SUMMARIES_INPATH)

    topics = pd.read_pickle(TOPICS_INPATH)
    probs = np.load(PROBS_INPATH)
    representative_docs = pd.read_pickle(REPRESENTATIVE_DOCS_INPATH)

    df_vis = create_df_for_viz(embeddings, topic_model, topics, docs, seed=SEED)

    df_vis = df_vis.merge(
        scite_abstracts, left_on="doc", right_on="title_abstract", how="left"
    )

    df_vis = df_vis.merge(
        cluster_summaries, left_on="topic", right_on="topic", how="left"
    )

    basic_chart(df_vis)

    chart_scaled_by_citations(df_vis)
