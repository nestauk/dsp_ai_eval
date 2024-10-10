from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
    PartOfSpeech,
)
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langfuse.callback import CallbackHandler

import re
import pandas as pd
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time
from umap import UMAP
from dotenv import load_dotenv
import os
from typing import List
from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils import utils
from tqdm import tqdm
from datetime import date
from time import sleep

from langfuse.callback import CallbackHandler

logger = logging.getLogger(__name__)
langfuse_handler = CallbackHandler()

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GPT_MODEL = config["summarization_pipeline"]["gpt_model"]


def get_openai_model(
    model=config["summarization_pipeline"]["gpt_model"],
) -> OpenAI:
    from langfuse.openai import openai

    client = openai.OpenAI()
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    topic: <topic label>
    """
    openai_model = OpenAI(
        client, model=model, exponential_backoff=True, chat=True, prompt=prompt
    )
    return openai_model


def get_anthropic_model(
    model=config["summarization_pipeline"]["gpt_model"],
):
    from bertopic.representation import LangChain

    pass


def create_new_topic_model(
    hdbscan_min_cluster_size=10,
    tfidf_min_df=2,
    tfidf_max_df=0.9,
    tfidf_ngram_range=(1, 2),
    seed=config["seed"],
    calculate_probabilities=False,
    embedding_model=config["embedding_model"],
):
    # Check if 'en_core_web_sm' is installed, if not, download it
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading...")
        os.system("python -m spacy download en_core_web_sm")
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
    logger.info(f"Min df: {tfidf_min_df}")
    logger.info(f"Max df: {tfidf_max_df}")

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # GPT-3.5
    openai_model = get_openai_model()

    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        "POS": pos_model,
    }

    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
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


def create_df_for_viz(
    embeddings, topic_model: BERTopic, topics, docs, seed=42, n_components=2
):
    umap_2d = UMAP(random_state=seed, n_components=n_components)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = topic_model.get_topic_info()[["Topic", "Name"]]

    # Create a DataFrame for visualization
    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    return df_vis


def get_top_docs_per_topic(
    abstracts_df: pd.DataFrame, topics: List[str], docs, probs, n_docs=10, axis=1
):
    df = pd.DataFrame(
        {
            "DocID": range(len(docs)),
            "Topic": topics,
            "Probability": probs.max(axis=axis),
        }
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
        top_docs: pd.DataFrame = abstracts_df[
            (abstracts_df["Topic"] == topic) & (abstracts_df["Probability"] == 1.0)
        ]

        if len(top_docs) < n_docs:
            top_docs = (
                abstracts_df[abstracts_df["Topic"] == topic]
                .sort_values(by="Probability", ascending=False)
                .head(n_docs)
            )
        else:
            cite_col = (
                "cited_by_count"
                if "total_cites" not in abstracts_df.columns
                else "total_cites"
            )
            top_docs = top_docs.sort_values(by=cite_col, ascending=False).head(n_docs)

        top_docs_per_topic[topic] = top_docs

    return abstracts_df, top_docs_per_topic


# Define the structure for the output we want to get back from GPT
class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


def get_summaries(
    topic_model: BERTopic,
    top_docs_per_topic,
    trace_name: str,
    text_col="answer_cleaned",
    model_name=GPT_MODEL,
    temperature=config["summarization_pipeline"]["gpt_temp"],
):
    summary_info = topic_model.get_topic_info()
    # Filter out the noise cluster (on the assumption that it's not possible to get a helpful summary of this very sparse, noisy cluster)
    summary_info = summary_info[summary_info["Topic"] != -1]

    summaries = {}

    for topic in tqdm(summary_info["Topic"]):
        text_list = top_docs_per_topic[topic][text_col].to_list()
        texts = "\n".join(text_list)
        keyword_list = summary_info[summary_info["Topic"] == topic]["KeyBERT"].tolist()[
            0
        ]
        keywords = ", ".join(keyword_list)

        llm = ChatOpenAI(
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

        langfuse_handler = CallbackHandler(
            user_id="ohid-maintaining-weightloss",
            session_id=f"{date.today().isoformat()}",
            trace_name=trace_name,
        )

        # Generate the summary
        summary_result = llm_chain.invoke(
            {"texts": texts, "keywords": keywords},
            config={"callbacks": [langfuse_handler]},
        )
        output = parser.invoke(summary_result, config={"callbacks": [langfuse_handler]})

        summaries[topic] = {
            "Name:": output.name,
            "Description:": output.description,
            "Docs": text_list,
            "Keywords": keyword_list,
            "Model": GPT_MODEL,
            "Temperature": temperature,
            "Prompt": summary_prompt,
        }
        sleep(2)  # Add a delay - avoids the error "HTTP/1.1 429 Too Many Requests"

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


def reclustering_model_from_topics(
    zeroshot_topic_list: List[str],
    zeroshot_min_similarity: float = config["oa_reclustering_pipeline"][
        "zeroshot_min_similarity"
    ],
    embedding_model=config["embedding_model"],
    calculate_probabilities=False,
    seed=config["seed"],
    hdbscan_min_cluster_size=5,
    tfidf_min_df=2,
    tfidf_max_df=0.9,
    tfidf_ngram_range=(1, 2),
    min_topic_size=config["oa_reclustering_pipeline"]["min_topic_size"],
):

    umap_model = UMAP(
        n_neighbors=3,
        n_components=10,
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

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")

    # GPT-3.5
    openai_model = get_openai_model()

    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "POS": pos_model,
    }

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
        calculate_probabilities=calculate_probabilities,
        # Hyperparameters
        min_topic_size=min_topic_size,
        top_n_words=3,
        verbose=False,
    )

    return topic_model
