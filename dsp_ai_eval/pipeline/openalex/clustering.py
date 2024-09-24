from dsp_ai_eval.utils import openalex_config

from typing import List

import typer

app = typer.Typer()

# all configs loaded here
config = openalex_config.get_all_configs()


@app.command()
def cluster_abstracts():
    from dsp_ai_eval.pipeline.openalex import cluster_abstracts

    cluster_abstracts.run_pipeline(config=config)


@app.command()
def summarise_clusters():
    from dsp_ai_eval.pipeline.openalex import cluster_summarization
    from dsp_ai_eval.pipeline.openalex import clean_cluster_summaries

    cluster_summarization.run_pipeline(config=config)
    clean_cluster_summaries.run_pipeline(config=config)


@app.command()
def create_plots():
    from dsp_ai_eval.pipeline.openalex import plot_abstract_clusters

    plot_abstract_clusters.run_pipeline(config=config)


@app.command()
def recluster():
    import questionary
    from dsp_ai_eval.getters.openalex import get_cluster_summaries_clean
    from dsp_ai_eval.pipeline.openalex import reclustering

    # summaries contain the topic Name
    cluster_summaries = get_cluster_summaries_clean("abstracts_pipeline")

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
