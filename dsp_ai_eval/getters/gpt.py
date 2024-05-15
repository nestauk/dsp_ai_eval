from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, config
from dsp_ai_eval.getters.utils import (
    load_s3_jsonl,
    load_s3_data,
    download_directory_from_s3,
)

EMBEDDING_MODEL = SentenceTransformer(config["embedding_model"])


def get_gpt_themes():
    filename = config["gpt_themes_pipeline"]["path_raw_data"]
    return load_s3_jsonl(
        S3_BUCKET,
        s3_file_name=filename,
        local_file=PROJECT_DIR / filename,
    )


def get_gpt_themes_cleaned():
    return load_s3_data(S3_BUCKET, config["gpt_themes_pipeline"]["path_cleaned_data"])


def get_gpt_themes_embeddings():
    return load_s3_data(
        S3_BUCKET, config["gpt_themes_pipeline"]["path_cleaned_data_w_embeddings"]
    )


def get_topic_model():
    download_directory_from_s3(
        S3_BUCKET,
        config["gpt_themes_pipeline"]["dir_topic_model"],
        PROJECT_DIR / config["gpt_themes_pipeline"]["dir_topic_model"],
    )
    return BERTopic.load(
        PROJECT_DIR / config["gpt_themes_pipeline"]["dir_topic_model"],
        embedding_model=EMBEDDING_MODEL,
    )


def get_topics():
    return load_s3_data(S3_BUCKET, config["gpt_themes_pipeline"]["path_topics"])


def get_probs():
    return load_s3_data(S3_BUCKET, config["gpt_themes_pipeline"]["path_probs"])


def get_representative_docs():
    return load_s3_data(S3_BUCKET, config["gpt_themes_pipeline"]["path_repr_docs"])


def get_cluster_summaries():
    return load_s3_data(S3_BUCKET, config["gpt_themes_pipeline"]["path_summaries"])


def get_cluster_summaries_cleaned():
    return load_s3_data(
        S3_BUCKET, config["gpt_themes_pipeline"]["path_summaries_cleaned"]
    )
