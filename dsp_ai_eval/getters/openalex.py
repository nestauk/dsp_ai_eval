from bertopic import BERTopic
import pandas as pd

from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, config, logger
from dsp_ai_eval.getters.utils import (
    load_s3_data,
    download_directory_from_s3,
)

rq_prefix: str = config["rq_prefix"]

# TODO: modify getters to default to a certain config, but allow str input for path


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
    df: pd.DataFrame = load_s3_data(
        S3_BUCKET,
        f"{rq_prefix}/{config['oa_abstracts_pipeline']['path_raw_data']}",
    )
    logger.info(f"Total number of works: {len(df)}")
    return df


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
    return load_s3_data(
        S3_BUCKET,
        f"{rq_prefix}/{filename}",
    )


def get_topic_model():
    dir_name = config["oa_abstracts_pipeline"]["dir_topic_model"]
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
    filemame = config["oa_abstracts_pipeline"]["path_topics"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filemame}")


def get_probs():
    filename = config["oa_abstracts_pipeline"]["path_probs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_representative_docs():
    filename = config["oa_abstracts_pipeline"]["path_repr_docs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries():
    filename = config["oa_abstracts_pipeline"]["path_summaries"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries_clean():
    filename = config["oa_abstracts_pipeline"]["path_summaries_cleaned"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_vis_data():
    filename = config["oa_abstracts_pipeline"]["path_vis_data"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")
