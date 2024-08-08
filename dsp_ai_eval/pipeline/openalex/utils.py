import pandas as pd
from typing import List

import tiktoken
from langchain_core.documents import Document

from dsp_ai_eval import logger
from dsp_ai_eval.utils import text_cleaning as tc


def filter_relevance_score(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    df = df[df["relevance_score"] > threshold]
    logger.info(
        f"Number of works remaining after filtering for relevance_score > {threshold}: {len(df)}"
    )
    return df


def min_cites_count(df: pd.DataFrame) -> int:
    works_with_min_cites = len(df[df["cited_by_count"] == df["cited_by_count"].min()])
    logger.info(
        f"Number of works with minimum number of citations: {works_with_min_cites}"
    )
    return works_with_min_cites


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def unnest_works(df: pd.DataFrame) -> pd.DataFrame:
    # unnest some columns for desired data
    biblio = pd.json_normalize(df["biblio"])
    pmid = pd.json_normalize(df["ids"])[["pmid"]]
    primary_location = pd.json_normalize(df["primary_location"]).rename(
        columns={
            "source.display_name": "journal",
            "source.host_organization_name": "publisher",
            "source.issn_l": "issn_l",
        }
    )[
        ["journal", "publisher", "issn_l"]
    ]  # issn_l, source.display_name, source.host_organization_name
    primary_topics = pd.json_normalize(df["primary_topic"]).rename(
        columns=lambda x: f"primary_topic.{x}"
    )

    df = pd.concat([df, biblio, pmid, primary_location, primary_topics], axis=1)
    return df


def clean_works(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "doi",
            "volume",
            "issue",
            "first_page",
            "last_page",
            "pmid",
            "journal",
            "publisher",
            "issn_l",
        ]
    )
    logger.info(
        f"Number of works remaining after dropping NA values on a subset: {len(df)}"
    )

    df = df.drop_duplicates("doi")
    logger.info(f"Number of works remaining after deduplicating on doi: {len(df)}")
    df = df.drop_duplicates("title")
    logger.info(f"Number of works remaining after deduplicating on titles: {len(df)}")

    df["abstract_clean"] = df["abstract"].apply(tc.clean_abstract)

    df = tc.clean_title_and_abstract(
        df, "abstract_clean", "title"
    )  # adds title_abstract column

    # count tokens in title_abstract
    df["num_tokens"] = df["title_abstract"].apply(num_tokens_from_string)
    print(
        "Number of documents more than 20000 tokens", len(df[df["num_tokens"] > 20000])
    )

    # remove retracted papers
    df = df[df["is_retracted"] == False]
    logger.info(
        f"Number of works remaining after filtering out retracted papers: {len(df)}"
    )

    return df


def bm25_filter(df: pd.DataFrame, rq: str, topk: int) -> pd.DataFrame:
    from dsp_ai_eval.getters.utils import get_bm25_scores

    df["bm25_score"] = get_bm25_scores(rq, df["title_abstract"].tolist())

    df = df.sort_values(by="bm25_score", ascending=False).head(topk)

    return df


def lc_docs_to_df(documents: List[Document]) -> pd.DataFrame:
    df_data = []

    for doc in documents:
        doc_data = doc.metadata
        doc_data["title_abstract"] = doc.page_content
        df_data.append(doc_data)

    return pd.DataFrame(df_data)


def lc_bm25_filter(df: pd.DataFrame, rq: str, topk: int) -> pd.DataFrame:
    from dsp_ai_eval.getters.utils import get_bm25_scores
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.document_loaders import DataFrameLoader

    df["bm25_score"] = get_bm25_scores(rq, df["title_abstract"].tolist())

    loader = DataFrameLoader(df, page_content_column="title_abstract")
    retriever = BM25Retriever.from_documents(documents=loader.load(), k=topk)
    res_df = lc_docs_to_df(retriever.invoke(rq))

    return res_df


def embed_works(
    sentences: List[str], model_name: str = "all-miniLM-L6-v2", n_batches: int = 10
) -> List:
    from more_itertools import chunked
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    batches = list(chunked(sentences, n=n_batches))
    logger.info(f"Number of chunks: {len(batches)}")

    embeddings = []

    for batch in batches:
        vecs = model.encode(
            batch,
            show_progress_bar=True,
        )
        embeddings.extend(vecs)

    return [vec.tolist() for vec in embeddings]
