import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils.clustering_utils import create_new_topic_model

EMBEDDING_MODEL = SentenceTransformer(config["embedding_model"])
ABSTRACTS_EMBEDDINGS_INPATH = (
    PROJECT_DIR / "inputs/data/embeddings/scite_embeddings.parquet"
)
TOPICS_OUTPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_topics.pkl"
PROBS_OUTPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_probs.npy"
MODEL_OUTPATH = (
    PROJECT_DIR / "outputs/models/bertopic_abstracts_model"
)  # This creates a folder, so you don't need to specify a file extension
REPRESENTATIVE_DOCS_OUTPATH = (
    PROJECT_DIR / "outputs/data/bertopic_abstracts_representative_docs.pkl"
)
SEED = config["seed"]

if __name__ == "__main__":
    scite_abstracts = pd.read_parquet(ABSTRACTS_EMBEDDINGS_INPATH)

    # Prepare docs and embeddings to input into BERTopic
    docs = scite_abstracts["title_abstract"].to_list()
    embeddings = scite_abstracts["embeddings"].apply(pd.Series).values

    # Initialise model with desired hyperparameters
    logging.info("Initialising BERTopic model...")
    topic_model = create_new_topic_model(
        hdbscan_min_cluster_size=30,
        tfidf_ngram_range=(1, 3),
        seed=SEED,
        calculate_probabilities=True,
    )

    # Train model
    logging.info("Training BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Save the topics and probability arrays
    logging.info("Saving topics and probabilities...")
    with open(TOPICS_OUTPATH, "wb") as f:
        pickle.dump(topics, f)

    np.save(PROBS_OUTPATH, probs)

    # Save the model itself
    logging.info("Saving topic model...")
    topic_model.save(
        MODEL_OUTPATH,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL,
    )

    # Save the representative documents
    logging.info("Saving representative documents...")
    representative_docs = topic_model.get_representative_docs()
    with open(REPRESENTATIVE_DOCS_OUTPATH, "wb") as f:
        pickle.dump(representative_docs, f)
