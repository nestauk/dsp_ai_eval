from bertopic import BERTopic
import pandas as pd


from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, logger, config
from dsp_ai_eval.getters.utils import (
    load_s3_data,
    download_directory_from_s3,
)

rq_prefix: str = config["rq_prefix"]


def get_openalex_works(
    user: str,
    research_question: str,
    min_cites: str = ">4",
    n_works: int = 10000,
) -> pd.DataFrame:
    import pyalex

    # set pyalex config email
    pyalex.config["email"] = user

    query = pyalex.Works().search(research_question).filter(cited_by_count=min_cites)

    results = []
    for page in query.paginate(per_page=200, n_max=n_works):
        results.extend(page)

    for page in results:
        page["abstract"] = page["abstract"]

    df = (
        pd.DataFrame(results)
        .dropna(subset=["title", "abstract"])
        .drop(columns=["abstract_inverted_index"])
    )

    return df


def get_works_raw() -> pd.DataFrame:
    df = load_s3_data(
        S3_BUCKET,
        f"{rq_prefix}/{config['oa_abstracts_pipeline']['path_raw_data']}",
    )
    if isinstance(df, pd.DataFrame):
        logger.info(f"Total number of works: {len(df)}")
        return df
    else:
        raise ValueError("Error loading raw works data, configured path is invalid")


def get_works_filtered():
    return load_s3_data(
        S3_BUCKET,
        f"{rq_prefix}/{config['oa_abstracts_pipeline']['path_filtered_data']}",
    )


def get_works_bm25_filtered():
    return load_s3_data(
        S3_BUCKET,
        f"{rq_prefix}/{config['oa_abstracts_pipeline']['path_bm25_filtered_data']}",
    )


def get_openalex_df_w_embeddings():
    filename = config["oa_abstracts_pipeline"]["path_cleaned_data_w_embeddings"]

    data = load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")

    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)


def get_topic_model(pipeline: str) -> BERTopic:

    if pipeline == "openalex_abstracts":
        dir_name = config["oa_abstracts_pipeline"]["dir_topic_model"]
    elif pipeline == "reclustering":
        dir_name = config["reclustering_pipeline"]["dir_topic_model"]
    dl_path = download_directory_from_s3(
        S3_BUCKET,
        f"{rq_prefix}/{dir_name}",
        PROJECT_DIR / dir_name,
    )
    return BERTopic.load(
        dl_path.as_posix(),
        embedding_model=config["embedding_model"],
    )


def get_topics(pipeline: str):

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_topics"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_topics"]

    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_probs(pipeline: str):

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_probs"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_probs"]

    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_representative_docs(pipeline: str):

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_repr_docs"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_repr_docs"]

    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries(pipeline: str):

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_summaries"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_summaries"]

    data = load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")

    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)


def get_cluster_summaries_clean(pipeline: str):

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_summaries_cleaned"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_summaries_cleaned"]

    data = load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")

    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)


def get_vis_data(pipeline: str) -> pd.DataFrame:

    if pipeline == "openalex_abstracts":
        filename = config["oa_abstracts_pipeline"]["path_vis_data"]
    elif pipeline == "reclustering":
        filename = config["reclustering_pipeline"]["path_vis_data"]

    data = load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")

    if isinstance(data, pd.DataFrame):
        return data
    else:
        return pd.DataFrame(data)
