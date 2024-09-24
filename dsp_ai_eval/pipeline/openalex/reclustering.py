from pathlib import Path
from typing import Optional, Union, List

from dsp_ai_eval import PROJECT_DIR, S3_BUCKET, logger, config


def run_pipeline(topic: int, keywords: List[str], config=config):
    from dsp_ai_eval.utils.clustering_utils import (
        reclustering_model_from_topics,
        get_top_docs_per_topic,
        get_summaries,
        clean_cluster_summaries,
    )
    from dsp_ai_eval.getters.openalex import get_vis_data
    from dsp_ai_eval.getters.utils import save_to_s3, copy_folder_to_s3
    from dsp_ai_eval.pipeline.openalex.plot_abstract_clusters import (
        create_df_for_viz,
        create_chart,
    )
    import pandas as pd

    df = (
        get_vis_data("openalex_abstracts")
        .query("Topic == @topic")
        .drop(
            columns=[
                "x",
                "y",
                "topic",
                "Topic",
                "Name",
                "doc",
                #'total_cites',
                "topic_name",
                "topic_description",
                "representative_docs",
                "topic_keywords",
            ]
        )
    )

    rq_prefix = config["rq_prefix"]

    ZEROSHOT_MIN_SIMILARITY = config["oa_reclustering_pipeline"][
        "zeroshot_min_similarity"
    ]
    EMBEDDING_MODEL = config["embedding_model"]

    TOPICS_OUTPATH = config["oa_reclustering_pipeline"]["path_topics"]
    PROBS_OUTPATH = config["oa_reclustering_pipeline"]["path_probs"]
    MODEL_OUTPATH = config["oa_reclustering_pipeline"]["dir_topic_model"]
    REPRESENTATIVE_DOCS_OUTPATH = config["oa_reclustering_pipeline"]["path_repr_docs"]

    # Prepare docs and embeddings to input into BERTopic
    docs = df["title_abstract"].tolist()
    embeddings = df["embeddings"].apply(pd.Series).values

    # Initialise model with desired hyperparameters
    topic_model = reclustering_model_from_topics(
        zeroshot_topic_list=keywords,
        zeroshot_min_similarity=ZEROSHOT_MIN_SIMILARITY,
        embedding_model=EMBEDDING_MODEL,
    )

    # Train model
    logger.info("Training BERTopic model...")
    topics, _ = topic_model.fit_transform(docs)
    topics, probs = topic_model.transform(docs)

    # Save the topics and probability arrays
    logger.info("Saving topics and probabilities...")
    save_to_s3(S3_BUCKET, topics, f"{rq_prefix}/{TOPICS_OUTPATH}")

    save_to_s3(S3_BUCKET, probs, f"{rq_prefix}/{PROBS_OUTPATH}")

    # Save the model itself
    logger.info("Saving topic model...")
    topic_model.save(
        PROJECT_DIR / MODEL_OUTPATH,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL,
    )
    copy_folder_to_s3(
        PROJECT_DIR / MODEL_OUTPATH, S3_BUCKET, f"{rq_prefix}/{MODEL_OUTPATH}"
    )

    # Save the representative documents
    logger.info("Saving representative documents...")
    representative_docs = topic_model.get_representative_docs()
    save_to_s3(
        S3_BUCKET, representative_docs, f"{rq_prefix}/{REPRESENTATIVE_DOCS_OUTPATH}"
    )

    # summarise reclustered topic
    SUMMARIES_OUTPATH = config["oa_reclustering_pipeline"]["path_summaries"]

    _, top_docs_per_topic = get_top_docs_per_topic(
        abstracts_df=df, topics=topics, docs=docs, probs=probs, n_docs=5, axis=None
    )

    summaries = get_summaries(
        topic_model,
        top_docs_per_topic,
        trace_name="abstract_summary",
        text_col="title_abstract",
    )

    save_to_s3(S3_BUCKET, summaries, f"{rq_prefix}/{SUMMARIES_OUTPATH}")

    # clean reclustered topic summaries
    CLUSTER_SUMMARIES_OUTPATH = config["oa_reclustering_pipeline"][
        "path_summaries_cleaned"
    ]
    cluster_summaries = clean_cluster_summaries(summaries)
    save_to_s3(S3_BUCKET, cluster_summaries, f"{rq_prefix}/{CLUSTER_SUMMARIES_OUTPATH}")

    # plot reclustered topic
    df_vis = create_df_for_viz(
        embeddings, topic_model, topics, docs, seed=config["seed"]
    )

    df_vis = df_vis.merge(df, left_on="doc", right_on="title_abstract", how="left")

    df_vis = df_vis.merge(
        cluster_summaries, left_on="topic", right_on="topic", how="left"
    )

    df_vis["topic_name"].fillna("NA", inplace=True)

    save_to_s3(
        S3_BUCKET,
        df_vis,
        f"{config['rq_prefix']}/{config['oa_reclustering_pipeline']['path_vis_data']}",
    )

    create_chart(
        df_vis,
        scale_by_citations=False,
        filename_suffix=f"_{topic}_reclustering",
        add_topic_legend=True,
    )

    create_chart(
        df_vis,
        scale_by_citations=True,
        filename_suffix=f"_{topic}_reclustering_citations",
        add_topic_legend=True,
    )


if __name__ == "__main__":
    run_pipeline()
