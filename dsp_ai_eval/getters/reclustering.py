from bertopic import BERTopic

from dsp_ai_eval import S3_BUCKET, PROJECT_DIR, config, logger
from dsp_ai_eval.getters.utils import (
    load_s3_data,
    download_directory_from_s3,
)

rq_prefix: str = config["rq_prefix"]


def get_topic_model():
    dir_name = config["oa_reclustering_pipeline"]["dir_topic_model"]
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
    filename = config["oa_reclustering_pipeline"]["path_topics"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_probs():
    filename = config["oa_reclustering_pipeline"]["path_probs"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")


def get_cluster_summaries_clean():
    filename = config["oa_reclustering_pipeline"]["path_summaries_cleaned"]
    return load_s3_data(S3_BUCKET, f"{rq_prefix}/{filename}")
