from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, config
from dsp_ai_eval.getters.utils import (
    load_s3_jsonl,
    load_s3_data,
    download_directory_from_s3,
)

rq_prefix: str = config["rq_prefix"]


def get_gpt_themes():
    filename = config["gpt_themes_pipeline"]["path_raw_data"]
    return load_s3_jsonl(
        S3_BUCKET,
        s3_file_name=f"{rq_prefix}/{filename}",
        local_file=PROJECT_DIR / filename,
    )


def get_gpt_themes_cleaned():
    filename = config["gpt_themes_pipeline"]["path_cleaned_data"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_gpt_themes_embeddings():
    filename = config["gpt_themes_pipeline"]["path_cleaned_data_w_embeddings"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_topic_model():
    dir_name = config["gpt_themes_pipeline"]["dir_topic_model"]
    EMBEDDING_MODEL = SentenceTransformer(
        config["embedding_model"], trust_remote_code=True, truncate_dim=384
    )
    download_directory_from_s3(
        S3_BUCKET,
        f"{rq_prefix}/{dir_name}",
        PROJECT_DIR / dir_name,
    )
    return BERTopic.load(
        PROJECT_DIR / dir_name,
        embedding_model=EMBEDDING_MODEL,
    )


def get_topics():
    filename = config["gpt_themes_pipeline"]["path_topics"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_probs():
    filename = config["gpt_themes_pipeline"]["path_probs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_representative_docs():
    filename = config["gpt_themes_pipeline"]["path_repr_docs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries():
    filename = config["gpt_themes_pipeline"]["path_summaries"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries_cleaned():
    filename = config["gpt_themes_pipeline"]["path_summaries_cleaned"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")
