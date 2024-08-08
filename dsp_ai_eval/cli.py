from dsp_ai_eval.pipeline import openalex

import typer

app = typer.Typer()

app.add_typer(openalex.app, name="openalex")

# TODO: function to build a config

if __name__ == "__main__":
    app()
