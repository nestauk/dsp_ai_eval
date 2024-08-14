from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, config, logging
from dsp_ai_eval.getters.utils import (
    load_s3_data,
    download_directory_from_s3,
)

N_MOST_RELEVANT_PAPERS = config["abstracts_pipeline"]["n_most_relevant_papers"]

rq_prefix: str = config["rq_prefix"]


def read_scite_abstracts(
    filepath: str,
    category: str = "main",
    n_abstracts: int = N_MOST_RELEVANT_PAPERS,
    bucket=S3_BUCKET,
) -> pd.DataFrame:
    """
    Reads scientific paper abstracts from a CSV file, assigns a category to each, and filters for relevance.

    The reason for 'category' is that we have some 'core' papers from scite Assistant,
    plus the hoardes of papers that resulted from the scite searched. The core and wider
    papers are read in separately and then concatenated. After concatenation, the 'category' column
    tells us whether this paper was originally one of the core citations or not.

    We also filter the papers if they are from the wider searches, as scite orders results by
    relevance and so you can get the top 100/1000/10,000 most relevant papers as desired.

    Parameters:
    - filepath (str): The path to the CSV file containing the paper abstracts.
    - category (str, optional): The category to assign to each paper abstract. Defaults to "main".
    - n_abstracts (int, optional): The number of most relevant paper abstracts to return. Defaults to N_MOST_RELEVANT_PAPERS.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered set of paper abstracts, each assigned with the specified category.
    """
    df = load_s3_data(bucket, filepath)

    df["category"] = category

    # Scite by default sorts by relevance, so take only the N most relevant papers
    # (You can download up to 10,000! But that seems excessive)
    if len(df) > n_abstracts:
        df = df.head(n_abstracts)

    return df


def get_abstracts(n_abstracts=N_MOST_RELEVANT_PAPERS):
    scite_main_abstracts = read_scite_abstracts(
        f"{rq_prefix}/" + config["abstracts_pipeline"]["path_scite_core_references"],
        "main",
        n_abstracts,
    )
    scite_wider_abstracts1 = read_scite_abstracts(
        f"{rq_prefix}/" + config["abstracts_pipeline"]["path_scite_search1"],
        "wider",
        n_abstracts,
    )
    # scite_wider_abstracts2 = read_scite_abstracts(
    #     config["abstracts_pipeline"]["path_scite_search2"], "wider", n_abstracts
    # )

    scite_abstracts = pd.concat(
        [scite_main_abstracts, scite_wider_abstracts1]  # , scite_wider_abstracts2
    )

    return scite_abstracts


def get_scite_df_w_embeddings():
    filename = config["abstracts_pipeline"]["path_cleaned_data_w_embeddings"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_topic_model():
    dir_name = config["abstracts_pipeline"]["dir_topic_model"]
    dl_path = download_directory_from_s3(
        S3_BUCKET,
        f"{rq_prefix}/{dir_name}",
        PROJECT_DIR / dir_name,
    )
    return BERTopic.load(
        dl_path,
        embedding_model=config["embedding_model"],
    )


def get_topics():
    filemame = config["abstracts_pipeline"]["path_topics"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filemame}")


def get_probs():
    filename = config["abstracts_pipeline"]["path_probs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_representative_docs():
    filename = config["abstracts_pipeline"]["path_repr_docs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries():
    filename = config["abstracts_pipeline"]["path_summaries"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries_clean():
    filename = config["abstracts_pipeline"]["path_summaries_cleaned"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")
