from dsp_ai_eval import config, S3_BUCKET
from dsp_ai_eval.getters.utils import save_to_s3
from dsp_ai_eval.getters.openalex import get_openalex_works, get_works_raw

import typer
from typing import Any, Dict, Optional
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer()

user = "henry.nurick@bi.team"  # use a separate config file
RQ = config["RQ"]
rq_prefix = config["rq_prefix"]
OUTPATH = config["oa_abstracts_pipeline"]["path_raw_data"]


@app.command()
def get(
    config: Annotated[Optional[Path], typer.Option()] = None,
    user: str = user,
    research_question: str = RQ,
    min_cites: str = ">4",
    n_works: int = 10000,
):
    """Pipeline to get works from OpenAlex.

    Args:
        user (str): _description_
        research_question (str): _description_
        min_cites (str, optional): _description_. Defaults to ">4".
        n_works (int, optional): _description_. Defaults to 10000.
    """
    works_df = get_openalex_works(
        user=user,
        research_question=research_question,
        min_cites=min_cites,
        n_works=n_works,
    )

    save_to_s3(S3_BUCKET, works_df, f"{rq_prefix}/{OUTPATH}")


@app.command()
def process(
    config=config,
    # config: Annotated[Optional[Path], typer.Option()] = None,
    openalex_rmin: int = 10,
    bm25_topk: int = 1000,
):
    # config = eval(config)
    import pandas as pd
    from dsp_ai_eval.pipeline.openalex.utils import (
        filter_relevance_score,
        min_cites_count,
        unnest_works,
        clean_works,
        bm25_filter,
        embed_works,
    )

    df: pd.DataFrame = (
        get_works_raw()
        .pipe(filter_relevance_score, threshold=openalex_rmin)
        .pipe(unnest_works)
        .pipe(clean_works)
    )
    # TODO change to cleaned
    FILTERED_OUT_PATH = config["oa_abstracts_pipeline"]["path_filtered_data"]
    FILTERED_S3_KEY = f"{rq_prefix}/{FILTERED_OUT_PATH}"
    save_to_s3(S3_BUCKET, df, FILTERED_S3_KEY)

    filtered: pd.DataFrame = bm25_filter(df, rq=RQ, topk=bm25_topk)
    BM25_FILTERED_OUT_PATH = config["oa_abstracts_pipeline"]["path_bm25_filtered_data"]
    BM25_FILTERED_S3_KEY = f"{rq_prefix}/{BM25_FILTERED_OUT_PATH}"
    save_to_s3(S3_BUCKET, filtered, BM25_FILTERED_S3_KEY)

    filtered["embeddings"] = embed_works(filtered["title_abstract"].tolist())
    EMBEDDINGS_OUT_PATH = config["oa_abstracts_pipeline"][
        "path_cleaned_data_w_embeddings"
    ]
    EMBEDDINGS_S3_KEY = f"{rq_prefix}/{EMBEDDINGS_OUT_PATH}"
    save_to_s3(S3_BUCKET, filtered, EMBEDDINGS_S3_KEY)

    min_cites_count(filtered)


@app.command()
def run_pipeline(
    config=config,
    # config: Annotated[Optional[Path], typer.Option()] = None,
):
    get(config=config)
    process(config=config)


if __name__ == "__main__":
    app()
