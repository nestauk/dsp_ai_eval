from dsp_ai_eval import (
    S3_BUCKET,
    config,
)  # TODO: add S3 bucket to config, or use local path
from dsp_ai_eval.getters.utils import save_to_s3
from dsp_ai_eval.getters.openalex import get_openalex_works, get_works_raw

import typer

app = typer.Typer()

user = config["openalex_user"]
RQ = config["RQ"]
rq_prefix = config["rq_prefix"]
OUTPATH = config["oa_abstracts_pipeline"]["path_raw_data"]


@app.command()
def get(
    s3_bucket: str = S3_BUCKET,
    path_raw_data: str = OUTPATH,
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

    save_to_s3(
        bucket_name=s3_bucket,
        output_var=works_df,
        output_file_dir=f"{rq_prefix}/{path_raw_data}",
    )


@app.command()
def process(
    rq_prefix: str,
    path_filtered_data: str,
    path_bm25_filtered_data: str,
    path_cleaned_data_w_embeddings: str,
    s3_bucket: str,
    openalex_rmin: int = 10,
    bm25_topk: int = 1000,
):
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

    save_to_s3(
        bucket_name=s3_bucket,
        output_var=df,
        output_file_dir=f"{rq_prefix}/{path_filtered_data}",
    )

    filtered: pd.DataFrame = bm25_filter(df, rq=RQ, topk=bm25_topk)
    save_to_s3(
        bucket_name=s3_bucket,
        output_var=filtered,
        output_file_dir=f"{rq_prefix}/{path_bm25_filtered_data}",
    )

    filtered["embeddings"] = embed_works(filtered["title_abstract"].tolist())
    save_to_s3(
        bucket_name=s3_bucket,
        output_var=filtered,
        output_file_dir=f"{rq_prefix}/{path_cleaned_data_w_embeddings}",
    )

    min_cites_count(filtered)


@app.command()
def run_pipeline(
    path_raw_data: str = OUTPATH,
    rq_prefix: str = rq_prefix,
    path_filtered_data: str = config["oa_abstracts_pipeline"]["path_filtered_data"],
    path_bm25_filtered_data: str = config["oa_abstracts_pipeline"][
        "path_bm25_filtered_data"
    ],
    path_cleaned_data_w_embeddings: str = config["oa_abstracts_pipeline"][
        "path_cleaned_data_w_embeddings"
    ],
    user: str = user,
    s3_bucket: str = S3_BUCKET,
    research_question: str = RQ,
    min_cites: str = ">4",
    n_works: int = 10000,
    openalex_rmin: int = 10,
    bm25_topk: int = 1000,
):
    get(
        s3_bucket=s3_bucket,
        path_raw_data=path_raw_data,
        user=user,
        research_question=research_question,
        min_cites=min_cites,
        n_works=n_works,
    )

    process(
        rq_prefix=rq_prefix,
        path_filtered_data=path_filtered_data,
        path_bm25_filtered_data=path_bm25_filtered_data,
        path_cleaned_data_w_embeddings=path_cleaned_data_w_embeddings,
        s3_bucket=s3_bucket,
        openalex_rmin=openalex_rmin,
        bm25_topk=bm25_topk,
    )


if __name__ == "__main__":
    app()
