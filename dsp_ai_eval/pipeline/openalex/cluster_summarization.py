from dsp_ai_eval import config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic, get_summaries
from dsp_ai_eval.getters.openalex import (
    get_openalex_df_w_embeddings,
    get_topic_model,
    get_probs,
    get_topics,
)
from dsp_ai_eval.getters.utils import save_to_s3

from typing import Any, Dict


def run_pipeline(
    config: Dict[Any, Any] = config,
):

    SUMMARIES_OUTPATH = config["oa_abstracts_pipeline"]["path_summaries"]
    rq_prefix = config["rq_prefix"]
    pipeline_name = "abstracts_pipeline"

    df = get_openalex_df_w_embeddings()

    docs = df["title_abstract"].to_list()

    df, top_docs_per_topic = get_top_docs_per_topic(
        abstracts_df=df,
        topics=get_topics(pipeline=pipeline_name),
        docs=docs,
        probs=get_probs(pipeline=pipeline_name),
        n_docs=5,
    )

    topic_model = get_topic_model(pipeline=pipeline_name)

    summaries = get_summaries(
        topic_model,
        top_docs_per_topic,
        trace_name="abstract_summary",
        text_col="title_abstract",
    )

    save_to_s3(S3_BUCKET, summaries, f"{rq_prefix}/{SUMMARIES_OUTPATH}")


if __name__ == "__main__":
    run_pipeline()
