from dsp_ai_eval import PROJECT_DIR, S3_BUCKET, logger, config
from dsp_ai_eval.utils.clustering_utils import create_new_topic_model
from dsp_ai_eval.getters.openalex import get_openalex_df_w_embeddings
from dsp_ai_eval.getters.utils import save_to_s3, copy_folder_to_s3

from typing import Any, Dict


def run_pipeline(
    config: Dict[Any, Any] = config,
):
    import pandas as pd

    rq_prefix = config["rq_prefix"]
    TOPICS_OUTPATH = config["oa_abstracts_pipeline"]["path_topics"]
    PROBS_OUTPATH = config["oa_abstracts_pipeline"]["path_probs"]
    MODEL_OUTPATH = config["oa_abstracts_pipeline"]["dir_topic_model"]
    REPRESENTATIVE_DOCS_OUTPATH = config["oa_abstracts_pipeline"]["path_repr_docs"]
    HDBSCAN_MIN_CLUSTER_SIZE = config["oa_abstracts_pipeline"][
        "hdsbscan_min_cluster_size"
    ]
    TFIDF_NGRAM_MIN = config["oa_abstracts_pipeline"]["tfidf_ngram_min"]
    TFIDF_NGRAM_MAX = config["oa_abstracts_pipeline"]["tfidf_ngram_max"]

    df = get_openalex_df_w_embeddings()

    # Prepare docs and embeddings to input into BERTopic
    docs = df["title_abstract"].tolist()
    embeddings = df["embeddings"].apply(pd.Series).values

    # Initialise model with desired hyperparameters
    topic_model = create_new_topic_model(
        hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        tfidf_ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        seed=config["seed"],
        calculate_probabilities=True,
    )

    # Train model
    logger.info("Training BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

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
        save_embedding_model=config["embedding_model"],
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


if __name__ == "__main__":
    run_pipeline()
