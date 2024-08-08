from dsp_ai_eval import logging, config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic, get_summaries
from dsp_ai_eval.getters.gpt import (
    get_gpt_themes_embeddings,
    get_topics,
    get_probs,
    get_representative_docs,
    get_topic_model,
)
from dsp_ai_eval.getters.utils import save_to_s3

SUMMARIES_OUTPATH = config["gpt_themes_pipeline"]["path_summaries"]
rq_prefix: str = config["rq_prefix"]

if __name__ == "__main__":
    answers_long = get_gpt_themes_embeddings()
    logging.info(f"text data: {len(answers_long)}")

    docs = answers_long["answer_cleaned"].tolist()

    topic_model = get_topic_model()

    topics = get_topics()
    logging.info(f"topics: {len(answers_long)}")
    probs = get_probs()
    # representative_docs = get_representative_docs() # Not used

    answers_long, top_docs_per_topic = get_top_docs_per_topic(
        answers_long, topics, docs, probs, 5
    )

    summaries = get_summaries(topic_model, top_docs_per_topic, trace_name="gpt_summary")

    save_to_s3(S3_BUCKET, summaries, f"{rq_prefix}/{SUMMARIES_OUTPATH}")
