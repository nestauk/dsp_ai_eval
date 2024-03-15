from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
    PartOfSpeech,
)
from dotenv import load_dotenv
from hdbscan import HDBSCAN
import openai
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from dsp_ai_eval import PROJECT_DIR, logging
from dsp_ai_eval.utils import utils

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def create_new_topic_model(
    hdbscan_min_cluster_size=10,
    tfidf_min_df=2,
    tfidf_max_df=0.75,
    tfidf_ngram_range=(1, 2),
    gpt_model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    seed=42,
):
    sentence_model = SentenceTransformer("all-miniLM-L6-v2")

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        ngram_range=(1, 2),
    )

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # GPT-3.5
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    topic: <topic label>
    """
    openai_model = OpenAI(
        client, model=gpt_model, exponential_backoff=True, chat=True, prompt=prompt
    )

    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        "POS": pos_model,
    }

    topic_model = BERTopic(
        # Pipeline models
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=10,
        verbose=True,
    )

    return topic_model


def create_df_for_viz(embeddings, topic_model, topics, docs, seed=42):
    umap_2d = UMAP(random_state=seed)  # UMAP red
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = topic_model.get_topic_info()[["Topic", "Name"]]

    # Create a DataFrame for visualization
    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    return df_vis
