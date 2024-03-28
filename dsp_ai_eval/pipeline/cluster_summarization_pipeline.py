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

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic

embedding_model = SentenceTransformer(config["embedding_model"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = "gpt-4"
TEMP = 0.4

ABSTRACTS_EMBEDDINGS_INPATH = (
    PROJECT_DIR / "inputs/data/embeddings/scite_embeddings.parquet"
)
TOPICS_INPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_topics.pkl"
PROBS_INPATH = PROJECT_DIR / "outputs/data/bertopic_abstracts_model_probs.npy"
REPRESENTATIVE_DOCS_INPATH = (
    PROJECT_DIR / "outputs/data/bertopic_abstracts_representative_docs.pkl"
)

SUMMARIES_OUTPATH = PROJECT_DIR / "outputs/data/cluster_summaries.json"


# Define the structure for the output we want to get back from GPT
class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


if __name__ == "__main__":
    scite_abstracts = pd.read_parquet(ABSTRACTS_EMBEDDINGS_INPATH)

    docs = scite_abstracts["title_abstract"].to_list()
    embeddings = scite_abstracts["embeddings"].apply(pd.Series).values

    topic_model = BERTopic.load(
        PROJECT_DIR / "outputs/models/bertopic_abstracts_model",
        embedding_model=embedding_model,
    )

    topics = pd.read_pickle(TOPICS_INPATH)
    probs = np.load(PROBS_INPATH)
    representative_docs = pd.read_pickle(REPRESENTATIVE_DOCS_INPATH)

    scite_abstracts, top_docs_per_topic = get_top_docs_per_topic(
        scite_abstracts, topics, docs, probs, 5
    )

    summary_info = topic_model.get_topic_info()

    # Filter out the noise cluster (on the assumption that it's not possible to get a helpful summary of this very sparse, noisy cluster)
    summary_info = summary_info[summary_info["Topic"] != -1]

    summaries = {}

    for topic in summary_info["Topic"]:
        text_list = top_docs_per_topic[topic]["title_abstract"].to_list()
        texts = "\n".join(text_list)

        keyword_list = summary_info[summary_info["Topic"] == topic]["KeyBERT"].tolist()[
            0
        ]
        keywords = ", ".join(keyword_list)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL, temperature=TEMP
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

    with open(SUMMARIES_OUTPATH, "w") as f:
        json.dump(summaries, f, indent=4)
