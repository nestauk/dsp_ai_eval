from dsp_ai_eval import config, S3_BUCKET
from dsp_ai_eval.utils.clustering_utils import get_top_docs_per_topic, get_summaries
from dsp_ai_eval.getters.scite import (
    get_scite_df_w_embeddings,
    get_topic_model,
    get_probs,
    get_topics,
)
from dsp_ai_eval.getters.utils import save_to_s3

SUMMARIES_OUTPATH = config["abstracts_pipeline"]["path_summaries"]
rq_prefix = config["rq_prefix"]

if __name__ == "__main__":
    scite_abstracts = get_scite_df_w_embeddings()

    docs = scite_abstracts["title_abstract"].to_list()

    topic_model = get_topic_model()

    topics = get_topics()
    probs = get_probs()
    # representative_docs = get_representative_docs() # Not used

    scite_abstracts, top_docs_per_topic = get_top_docs_per_topic(
        scite_abstracts, topics, docs, probs, 5
    )

    summaries = get_summaries(
        topic_model,
        top_docs_per_topic,
        trace_name="abstract_summary",
        text_col="title_abstract",
    )

    save_to_s3(S3_BUCKET, summaries, f"{SUMMARIES_OUTPATH}")
