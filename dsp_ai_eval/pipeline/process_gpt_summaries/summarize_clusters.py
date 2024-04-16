from bertopic import BERTopic
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

from dsp_ai_eval import PROJECT_DIR, logging, config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic, get_summaries
from dsp_ai_eval.getters.gpt import (
    get_gpt_themes_cleaned,
    get_topics,
    get_probs,
    get_representative_docs,
    get_topic_model,
)
from dsp_ai_eval.getters.utils import download_directory_from_s3, save_to_s3

embedding_model = SentenceTransformer(config["embedding_model"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = config["summarization_pipeline"]["gpt_model"]
TEMP = config["summarization_pipeline"]["gpt_temp"]

# GPT_THEMES_INPATH = PROJECT_DIR / 'inputs/data/gpt/gpt_themes_repeats_cleaned_embeddings.csv'
# TOPICS_INPATH = PROJECT_DIR / "outputs/data/bertopic_gpt_themes_model_topics.pkl"
# PROBS_INPATH = PROJECT_DIR / "outputs/data/bertopic_gpt_themes_model_probs.npy"
# REPRESENTATIVE_DOCS_INPATH = (
#     PROJECT_DIR / "outputs/data/bertopic_gpt_themes_representative_docs.pkl"
# )

SUMMARIES_OUTPATH = config["gpt_themes_pipeline"]["path_summaries"]

if __name__ == "__main__":
    answers_long = get_gpt_themes_cleaned()

    docs = answers_long["answer_cleaned"].tolist()

    topic_model = get_topic_model()

    topics = get_topics()
    probs = get_probs()
    representative_docs = get_representative_docs()

    answers_long, top_docs_per_topic = get_top_docs_per_topic(
        answers_long, topics, docs, probs, 5
    )

    summaries = get_summaries(topic_model, top_docs_per_topic)

    save_to_s3(S3_BUCKET, summaries, SUMMARIES_OUTPATH)
