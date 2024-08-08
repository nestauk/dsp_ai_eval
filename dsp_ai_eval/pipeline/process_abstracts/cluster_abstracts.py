import pandas as pd
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import PROJECT_DIR, logging, config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import create_new_topic_model
from dsp_ai_eval.getters.scite import get_scite_df_w_embeddings
from dsp_ai_eval.getters.utils import save_to_s3, copy_folder_to_s3

EMBEDDING_MODEL = SentenceTransformer(config["embedding_model"])

rq_prefix = config["rq_prefix"]

TOPICS_OUTPATH = f'{rq_prefix}/{config["abstracts_pipeline"]["path_topics"]}'
PROBS_OUTPATH = f'{rq_prefix}/{config["abstracts_pipeline"]["path_probs"]}'
MODEL_OUTPATH = config["abstracts_pipeline"]["dir_topic_model"]
REPRESENTATIVE_DOCS_OUTPATH = (
    f'{rq_prefix}/{config["abstracts_pipeline"]["path_repr_docs"]}'
)
SEED = config["seed"]
HDBSCAN_MIN_CLUSTER_SIZE = config["abstracts_pipeline"]["hdsbscan_min_cluster_size"]
TFIDF_NGRAM_MIN = config["abstracts_pipeline"]["tfidf_ngram_min"]
TFIDF_NGRAM_MAX = config["abstracts_pipeline"]["tfidf_ngram_max"]

if __name__ == "__main__":
    scite_abstracts = get_scite_df_w_embeddings()

    # Prepare docs and embeddings to input into BERTopic
    docs = scite_abstracts["title_abstract"].to_list()
    embeddings = scite_abstracts["embeddings"].apply(pd.Series).values

    # Initialise model with desired hyperparameters
    logging.info("Initialising BERTopic model...")
    topic_model = create_new_topic_model(
        hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        tfidf_ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        seed=SEED,
        calculate_probabilities=True,
    )

    # Train model
    logging.info("Training BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Save the topics and probability arrays
    logging.info("Saving topics and probabilities...")
    save_to_s3(S3_BUCKET, topics, TOPICS_OUTPATH)

    save_to_s3(S3_BUCKET, probs, PROBS_OUTPATH)

    # Save the model itself
    logging.info("Saving topic model...")
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
    logging.info("Saving representative documents...")
    representative_docs = topic_model.get_representative_docs()
    save_to_s3(S3_BUCKET, representative_docs, REPRESENTATIVE_DOCS_OUTPATH)
