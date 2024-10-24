from dsp_ai_eval import S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic, get_summaries
from dsp_ai_eval.getters.openalex import (
    get_openalex_df_w_embeddings,
    get_topic_model,
    get_probs,
    get_topics,
)
from dsp_ai_eval.getters.utils import save_to_s3


def run_pipeline(
    rq_prefix: str,
    path_summaries: str,
    llm: str,
    temperature: float,
):
    pipeline_name = "openalex_abstracts"

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
        llm=llm,
        temperature=temperature,
    )

    save_to_s3(S3_BUCKET, summaries, f"{rq_prefix}/{path_summaries}")


if __name__ == "__main__":
    from dsp_ai_eval import config

    run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_summaries=config["oa_abstracts_pipeline"]["path_summaries"],
        llm=config["summarization_pipeline"]["gpt_model"],
        temperature=config["summarization_pipeline"]["gpt_temp"],
    )
