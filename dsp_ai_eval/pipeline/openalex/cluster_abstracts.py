from dsp_ai_eval import PROJECT_DIR, S3_BUCKET, logger, config
from dsp_ai_eval.utils.clustering_utils import create_new_topic_model
from dsp_ai_eval.getters.openalex import get_openalex_df_w_embeddings
from dsp_ai_eval.getters.utils import save_to_s3, copy_folder_to_s3


def run_pipeline(
    rq_prefix: str,
    path_topics: str,
    path_probs: str,
    dir_topic_model: str,
    path_repr_docs: str,
    hdbscan_min_cluster_size: int,
    tfidf_ngram_min: int,
    tfidf_ngram_max: int,
    seed: int,
    embedding_model: str,
    llm: str,
    umap_n_neighbors=15,
    umap_n_components=25,
):
    import pandas as pd

    df = get_openalex_df_w_embeddings()

    # Prepare docs and embeddings to input into BERTopic
    docs = df["title_abstract"].tolist()
    embeddings = df["embeddings"].apply(pd.Series).values

    try:
        # Initialise model with desired hyperparameters
        topic_model = create_new_topic_model(
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            tfidf_ngram_range=(tfidf_ngram_min, tfidf_ngram_max),
            seed=seed,
            calculate_probabilities=True,
            embedding_model=embedding_model,
            llm=llm,
            umap_n_neighbors=umap_n_neighbors,
            umap_n_components=umap_n_components,
        )

        # Train model
        logger.info("Training BERTopic model...")
        topics, probs = topic_model.fit_transform(docs, embeddings)
    except TypeError as e:
        logger.error(f"Error: {e}")
        topic_model = create_new_topic_model(
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            tfidf_ngram_range=(tfidf_ngram_min, tfidf_ngram_max),
            seed=seed,
            calculate_probabilities=True,
            embedding_model=embedding_model,
            llm=llm,
            umap_n_neighbors=umap_n_neighbors,
            umap_n_components=len(docs)
            - 2,  # n_components must be 2 less than n_samples or n_docs
        )

        # Train model
        logger.info("Training BERTopic model...")
        topics, probs = topic_model.fit_transform(docs, embeddings)

    # Save the topics and probability arrays
    logger.info("Saving topics and probabilities...")
    save_to_s3(S3_BUCKET, topics, f"{rq_prefix}/{path_topics}")

    save_to_s3(S3_BUCKET, probs, f"{rq_prefix}/{path_probs}")

    # Save the model itself
    logger.info("Saving topic model...")
    topic_model.save(
        PROJECT_DIR / dir_topic_model,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )
    copy_folder_to_s3(
        PROJECT_DIR / dir_topic_model, S3_BUCKET, f"{rq_prefix}/{dir_topic_model}"
    )

    # Save the representative documents
    logger.info("Saving representative documents...")
    representative_docs = topic_model.get_representative_docs()
    save_to_s3(S3_BUCKET, representative_docs, f"{rq_prefix}/{path_repr_docs}")


if __name__ == "__main__":
    from dsp_ai_eval import config

    run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_topics=config["oa_abstracts_pipeline"]["path_topics"],
        path_probs=config["oa_abstracts_pipeline"]["path_probs"],
        dir_topic_model=config["oa_abstracts_pipeline"]["dir_topic_model"],
        path_repr_docs=config["oa_abstracts_pipeline"]["path_repr_docs"],
        hdbscan_min_cluster_size=config["oa_abstracts_pipeline"][
            "hdbscan_min_cluster_size"
        ],
        tfidf_ngram_min=config["oa_abstracts_pipeline"]["tfidf_ngram_min"],
        tfidf_ngram_max=config["oa_abstracts_pipeline"]["tfidf_ngram_max"],
        seed=config["seed"],
        embedding_model=config["embedding_model"],
        llm=config["summarization_pipeline"]["gpt_model"],
    )
