from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
    PartOfSpeech,
)
from dotenv import load_dotenv
from hdbscan import HDBSCAN
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import numpy as np
import openai
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time
from umap import UMAP

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils import utils

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GPT_MODEL = config["summarization_pipeline"]["gpt_model"]
TEMP = config["summarization_pipeline"]["gpt_temp"]


def create_new_topic_model(
    hdbscan_min_cluster_size=10,
    tfidf_min_df=2,
    tfidf_max_df=0.9,
    tfidf_ngram_range=(1, 2),
    gpt_model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    seed=config["seed"],
    calculate_probabilities=False,
    embedding_model=config["embedding_model"],
):
    sentence_model = SentenceTransformer(embedding_model)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=50,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=tfidf_min_df,
        max_df=float(tfidf_max_df),
        ngram_range=tfidf_ngram_range,
    )
    logging.info(f"Min df: {tfidf_min_df}")
    logging.info(f"Max df: {tfidf_max_df}")

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
        calculate_probabilities=calculate_probabilities,
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


def get_top_docs_per_topic(abstracts_df, topics, docs, probs, n_docs=10):
    df = pd.DataFrame(
        {"DocID": range(len(docs)), "Topic": topics, "Probability": probs.max(axis=1)}
    )

    abstracts_df = pd.concat(
        [abstracts_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1
    )

    # For each topic, get the top 5 documents
    top_docs_per_topic = {}
    unique_topics = set(topics)
    for topic in unique_topics:
        if topic == -1:
            continue  # Skip the outlier topic
        top_docs = abstracts_df[
            (abstracts_df["Topic"] == topic) & (abstracts_df["Probability"] == 1.0)
        ]

        if len(top_docs) < n_docs:
            top_docs = (
                abstracts_df[abstracts_df["Topic"] == topic]
                .sort_values(by="Probability", ascending=False)
                .head(n_docs)
            )
        else:
            top_docs = top_docs.sort_values(by="total_cites", ascending=False).head(
                n_docs
            )

        top_docs_per_topic[topic] = top_docs

    return abstracts_df, top_docs_per_topic


# Define the structure for the output we want to get back from GPT
class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


def get_summaries(
    topic_model,
    top_docs_per_topic,
    text_col="answer_cleaned",
    openai_api_key=OPENAI_API_KEY,
    model_name=GPT_MODEL,
    temperature=TEMP,
):
    summary_info = topic_model.get_topic_info()
    logging.info(summary_info)
    # Filter out the noise cluster (on the assumption that it's not possible to get a helpful summary of this very sparse, noisy cluster)
    summary_info = summary_info[summary_info["Topic"] != -1]

    summaries = {}

    for topic in summary_info["Topic"]:
        text_list = top_docs_per_topic[topic][text_col].to_list()
        texts = "\n".join(text_list)
        logging.info(texts)
        keyword_list = summary_info[summary_info["Topic"] == topic]["KeyBERT"].tolist()[
            0
        ]
        keywords = ", ".join(keyword_list)

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
        )

        summary_prompt = """I have clustered some documents and pulled out the most representative documents for each cluster. Based on the documents given, and a list of keywords that are prominent in the cluster, please (a) give the cluster a name and (b) write a brief description of the cluster.
        \n
        DOCUMENTS:
        {texts}
        \n
        KEYWORDS:
        {keywords}
        \n
        {format_instructions}
        """

        # json_parser = JsonOutputParser()
        parser = PydanticOutputParser(pydantic_object=NameDescription)

        prompt = PromptTemplate(
            template=summary_prompt,
            input_variables=["texts", "keywords"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm_chain = prompt | llm

        # Generate the summary
        summary_result = llm_chain.invoke({"texts": texts, "keywords": keywords})
        output = parser.invoke(summary_result)

        summaries[topic] = {
            "Name:": output.name,
            "Description:": output.description,
            "Docs": text_list,
            "Keywords": keyword_list,
            "Model": GPT_MODEL,
            "Temperature": TEMP,
            "Prompt": summary_prompt,
        }
        time.sleep(2)  # Add a delay - avoids the error "HTTP/1.1 429 Too Many Requests"

    return summaries


def to_snake_case(s: str) -> str:
    # Replace all non-word characters (everything except letters and numbers) with an underscore
    s = re.sub(r"\W+", "_", s)
    # Convert to lowercase
    s = s.lower()
    # Remove leading and trailing underscores
    s = s.strip("_")
    return s


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names to snake case and strips leading or trailing punctuation.

    :param df: pandas DataFrame with any column names
    :return: pandas DataFrame with cleaned column names
    """
    new_columns = {col: to_snake_case(col) for col in df.columns}
    return df.rename(columns=new_columns)


def clean_cluster_summaries(cluster_summaries):
    # Convert the nested dictionary into a list of dictionaries
    data = [v for _, v in cluster_summaries.items()]

    # Create a DataFrame
    df = pd.DataFrame(data).reset_index().rename(columns={"index": "topic"})

    df = clean_column_names(df)

    df = df.rename(
        columns={
            "docs": "representative_docs",
            "name": "topic_name",
            "description": "topic_description",
            "keywords": "topic_keywords",
        }
    )

    return df[
        [
            "topic",
            "topic_name",
            "topic_description",
            "representative_docs",
            "topic_keywords",
        ]
    ]
