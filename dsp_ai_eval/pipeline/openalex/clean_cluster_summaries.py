from dsp_ai_eval import S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import clean_cluster_summaries
from dsp_ai_eval.getters.openalex import get_cluster_summaries
from dsp_ai_eval.getters.utils import save_to_s3


def run_pipeline(
    rq_prefix: str,
    path_summaries_cleaned: str,
):
    cluster_summaries_cleaned = get_cluster_summaries(
        pipeline="openalex_abstracts"
    ).pipe(clean_cluster_summaries)

    save_to_s3(
        S3_BUCKET, cluster_summaries_cleaned, f"{rq_prefix}/{path_summaries_cleaned}"
    )


if __name__ == "__main__":
    from dsp_ai_eval import config

    run_pipeline(
        config["rq_prefix"], config["oa_abstracts_pipeline"]["path_summaries_cleaned"]
    )
