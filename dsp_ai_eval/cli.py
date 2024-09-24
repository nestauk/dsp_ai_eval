from dsp_ai_eval.pipeline import openalex
from dsp_ai_eval.utils import openalex_config

import typer

app = typer.Typer()

app.add_typer(
    openalex.app,
    name="openalex",
)
app.add_typer(openalex_config.app, name="config")

if __name__ == "__main__":
    app()
