from dsp_ai_eval import config

from typing import List

import typer

app = typer.Typer()


@app.command()
def cluster_abstracts():
    from dsp_ai_eval.pipeline.openalex import cluster_abstracts

    cluster_abstracts.run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_topics=config["oa_abstracts_pipeline"]["path_topics"],
        path_probs=config["oa_abstracts_pipeline"]["path_probs"],
        dir_topic_model=config["oa_abstracts_pipeline"]["dir_topic_model"],
        path_repr_docs=config["oa_abstracts_pipeline"]["path_repr_docs"],
        hdbscan_min_cluster_size=config["oa_abstracts_pipeline"][
            "hdbscan_min_cluster_size"
        ],
        tfidf_ngram_min=config["oa_abstracts_pipeline"]["tfidf_ngram_min"],
        tfidf_ngram_max=config["oa_abstracts_pipeline"]["tfidf_ngram_max"],
        seed=config["seed"],
        embedding_model=config["embedding_model"],
        llm=config["summarization_pipeline"]["gpt_model"],
        umap_n_neighbors=config["oa_abstracts_pipeline"]["umap_n_neighbors"],
        umap_n_components=config["oa_abstracts_pipeline"]["umap_n_components"],
    )


@app.command()
def summarise_clusters():
    from dsp_ai_eval.pipeline.openalex import cluster_summarization
    from dsp_ai_eval.pipeline.openalex import clean_cluster_summaries

    cluster_summarization.run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_summaries=config["oa_abstracts_pipeline"]["path_summaries"],
        llm=config["summarization_pipeline"]["gpt_model"],
        temperature=config["summarization_pipeline"]["gpt_temp"],
    )
    clean_cluster_summaries.run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_summaries_cleaned=config["oa_abstracts_pipeline"][
            "path_summaries_cleaned"
        ],
    )


@app.command()
def create_plots():
    from dsp_ai_eval.pipeline.openalex import plot_abstract_clusters

    plot_abstract_clusters.run_pipeline(
        rq_prefix=config["rq_prefix"],
        path_vis_data=config["oa_abstracts_pipeline"]["path_vis_data"],
        seed=config["seed"],
    )


@app.command()
def recluster():
    import questionary
    from dsp_ai_eval.getters.openalex import get_cluster_summaries_clean
    from dsp_ai_eval.pipeline.openalex import reclustering

    # summaries contain the topic Name
    cluster_summaries = get_cluster_summaries_clean(pipeline="openalex_abstracts")

    topic = questionary.select(
        message="Select topic to recluster:",
        choices=[
            questionary.Choice(
                title=f"  {k}. {v}",
                value=k,
            )
            for k, v in cluster_summaries["topic_name"].to_dict().items()
        ],
    ).ask()
    typer.echo(f"Selected topic: {topic}")

    keywords_input: str = typer.prompt(
        "Enter (comma separated) keywords to recluster on", type=str
    )
    keywords: List[str] = keywords_input.replace(" ", "").split(",")

    reclustering.run_pipeline(config=config, topic=topic, keywords=keywords)


@app.command()
def run_pipeline():

    cluster_abstracts()
    summarise_clusters()
    create_plots()


if __name__ == "__main__":
    app()
