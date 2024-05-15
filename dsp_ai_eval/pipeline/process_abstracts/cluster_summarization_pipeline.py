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
from dsp_ai_eval.getters.scite import (
    get_scite_df_w_embeddings,
    get_topic_model,
    get_probs,
    get_representative_docs,
    get_topics,
)
from dsp_ai_eval.getters.utils import save_to_s3

embedding_model = SentenceTransformer(config["embedding_model"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = config["summarization_pipeline"]["gpt_model"]
TEMP = config["summarization_pipeline"]["gpt_temp"]

SUMMARIES_OUTPATH = config["abstracts_pipeline"]["path_summaries"]

if __name__ == "__main__":
    scite_abstracts = get_scite_df_w_embeddings()

    docs = scite_abstracts["title_abstract"].to_list()

    topic_model = get_topic_model()

    topics = get_topics()
    probs = get_probs()
    representative_docs = get_representative_docs()

    scite_abstracts, top_docs_per_topic = get_top_docs_per_topic(
        scite_abstracts, topics, docs, probs, 5
    )

    summaries = get_summaries(
        topic_model, top_docs_per_topic, text_col="title_abstract"
    )

    save_to_s3(S3_BUCKET, summaries, SUMMARIES_OUTPATH)
