from typing import Any, Dict, Optional
from typing_extensions import Annotated

from dsp_ai_eval import config
from dsp_ai_eval.pipeline.openalex import works
from dsp_ai_eval.pipeline.openalex import clustering

import typer
from pathlib import Path

app = typer.Typer()

app.add_typer(works.app, name="works")
app.add_typer(clustering.app, name="clustering")


@app.command()
def run_pipeline(config=config):
    # config: Annotated[Optional[Path], typer.Option()] = None
    config = eval(config)

    works.run_pipeline(config=config)
    clustering.run_pipeline(config=config)
