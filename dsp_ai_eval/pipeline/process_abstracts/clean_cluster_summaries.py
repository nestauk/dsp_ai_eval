from dsp_ai_eval import PROJECT_DIR, S3_BUCKET, config
from dsp_ai_eval.utils.clustering_utils import clean_cluster_summaries
from dsp_ai_eval.getters.scite import get_cluster_summaries
from dsp_ai_eval.getters.utils import save_to_s3

CLUSTER_SUMMARIES_OUTPATH = config["abstracts_pipeline"]["path_summaries_cleaned"]
rq_prefix = config["rq_prefix"]

if __name__ == "__main__":
    cluster_summaries = get_cluster_summaries()

    cluster_summaries_cleaned = clean_cluster_summaries(cluster_summaries)

    save_to_s3(S3_BUCKET, cluster_summaries_cleaned, f"{CLUSTER_SUMMARIES_OUTPATH}")
