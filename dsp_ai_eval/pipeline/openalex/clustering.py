from dsp_ai_eval import config as base_config

from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

import typer

app = typer.Typer()


@app.command()
def cluster_abstracts(config: Annotated[Optional[Path], typer.Option()] = None):
    from dsp_ai_eval.pipeline.openalex import cluster_abstracts

    cluster_abstracts.run_pipeline(config=config)


@app.command()
def summarise_clusters(config: Annotated[Optional[Path], typer.Option()] = None):
    from dsp_ai_eval.pipeline.openalex import cluster_summarization
    from dsp_ai_eval.pipeline.openalex import clean_cluster_summaries

    cluster_summarization.run_pipeline(config=config)
    clean_cluster_summaries.run_pipeline(config=config)


@app.command()
def create_plots(config: Annotated[Optional[Path], typer.Option()] = None):
    from dsp_ai_eval.pipeline.openalex import plot_abstract_clusters

    if config is None:
        config = base_config

    plot_abstract_clusters.run_pipeline(config=config)


@app.command()
def recluster(config: Annotated[Optional[Path], typer.Option()] = None):
    import questionary
    from dsp_ai_eval.getters.openalex import get_cluster_summaries_clean
    from dsp_ai_eval.pipeline.openalex import reclustering

    if config is None:
        config = base_config

    # summaries contain the topic Name
    cluster_summaries = get_cluster_summaries_clean()

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

    keywords: str = typer.prompt(
        "Enter (comma separated) keywords to recluster on", type=str
    )
    keywords: List[str] = keywords.replace(" ", "").split(",")

    reclustering.run_pipeline(config=config, topic=topic, keywords=keywords)


@app.command()
def run_pipeline(config: Annotated[Optional[Path], typer.Option()] = None):
    cluster_abstracts(config=config)
    summarise_clusters(config=config)
    create_plots(config=config)


if __name__ == "__main__":
    app()
