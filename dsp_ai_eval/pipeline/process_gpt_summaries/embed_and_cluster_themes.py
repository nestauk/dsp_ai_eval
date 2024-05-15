import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import PROJECT_DIR, logging, config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import create_new_topic_model
from dsp_ai_eval.getters.gpt import get_gpt_themes_cleaned
from dsp_ai_eval.getters.utils import save_to_s3, upload_file_to_s3, copy_folder_to_s3

model = SentenceTransformer(config["embedding_model"])
SEED = config["seed"]

GPT_THEMES_EMBEDDINGS_OUTPATH = config["gpt_themes_pipeline"][
    "path_cleaned_data_w_embeddings"
]
TOPICS_OUTPATH = config["gpt_themes_pipeline"]["path_topics"]
PROBS_OUTPATH = config["gpt_themes_pipeline"]["path_probs"]
MODEL_OUTPATH = config["gpt_themes_pipeline"]["dir_topic_model"]
REPRESENTATIVE_DOCS_OUTPATH = config["gpt_themes_pipeline"]["path_repr_docs"]

N_TOPICS = config["gpt_themes_pipeline"]["n_topics"]
HDBSCAN_MIN_CLUSTER_SIZE = config["gpt_themes_pipeline"]["hdsbscan_min_cluster_size"]
TFIDF_NGRAM_MIN = config["gpt_themes_pipeline"]["tfidf_ngram_min"]
TFIDF_NGRAM_MAX = config["gpt_themes_pipeline"]["tfidf_ngram_max"]

if __name__ == "__main__":
    answers_long = get_gpt_themes_cleaned()

    docs = answers_long["answer_cleaned"].tolist()
    embeddings = model.encode(docs, show_progress_bar=True)

    vectors_as_list = [list(vec) for vec in embeddings]

    # Add the embeddings as a column to the original dataframe
    answers_long["embeddings"] = vectors_as_list
    save_to_s3(
        S3_BUCKET, answers_long, GPT_THEMES_EMBEDDINGS_OUTPATH
    )  # answers_long.to_csv(GPT_THEMES_EMBEDDINGS_OUTPATH)

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

    # Reduce the number of topics after the initial model fitting
    topic_model.reduce_topics(docs, nr_topics=N_TOPICS)

    # Use transform to get the new topics and probabilities
    new_topics, new_probs = topic_model.transform(docs)

    # Save the topics and probability arrays
    logging.info("Saving topics and probabilities...")
    save_to_s3(S3_BUCKET, new_topics, TOPICS_OUTPATH)

    save_to_s3(S3_BUCKET, new_probs, PROBS_OUTPATH)

    # Save the model itself
    logging.info("Saving topic model...")
    topic_model.save(
        PROJECT_DIR / MODEL_OUTPATH,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=model,
    )

    copy_folder_to_s3(PROJECT_DIR / MODEL_OUTPATH, S3_BUCKET, MODEL_OUTPATH)

    # Save the representative documents
    logging.info("Saving representative documents...")
    representative_docs = topic_model.get_representative_docs()

    save_to_s3(S3_BUCKET, representative_docs, REPRESENTATIVE_DOCS_OUTPATH)
