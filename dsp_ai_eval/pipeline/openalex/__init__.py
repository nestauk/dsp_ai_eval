from dsp_ai_eval.pipeline.openalex import works
from dsp_ai_eval.pipeline.openalex import clustering

from dsp_ai_eval import config

import typer

app = typer.Typer()

user = config["openalex_user"]
RQ = config["RQ"]

# add subcommands
app.add_typer(works.app, name="works")
app.add_typer(clustering.app, name="clustering")


@app.command()
def run_pipeline(
    user: str = user,
    research_question: str = RQ,
    min_cites: str = ">4",
    n_works: int = 10000,
    openalex_rmin: int = 10,
    bm25_topk: int = 1000,
):
    works.run_pipeline(
        user=user,
        research_question=research_question,
        min_cites=min_cites,
        n_works=n_works,
        openalex_rmin=openalex_rmin,
        bm25_topk=bm25_topk,
    )
    clustering.run_pipeline()
