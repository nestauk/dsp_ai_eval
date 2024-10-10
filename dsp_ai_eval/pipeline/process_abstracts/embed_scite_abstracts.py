import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Union

from dsp_ai_eval import logging, config, S3_BUCKET
from dsp_ai_eval.utils import text_cleaning as tc
from dsp_ai_eval.getters.utils import save_to_s3
from dsp_ai_eval.getters.scite import get_abstracts

model = SentenceTransformer(
    config["embedding_model"], trust_remote_code=True, truncate_dim=384
)

OUT_PATH = config["abstracts_pipeline"]["path_cleaned_data_w_embeddings"]
rq_prefix = config["rq_prefix"]
S3_KEY = f"{OUT_PATH}"

CITATION_THRESHOLD = config["abstracts_pipeline"]["citation_threshold"]
N_MOST_RELEVANT_PAPERS = config["abstracts_pipeline"]["n_most_relevant_papers"]


def first_non_nan(series: pd.Series) -> Union[pd.Series, float]:
    return series.dropna().iloc[0] if not series.dropna().empty else np.nan


def create_agg_dict(df):
    """
    Create an aggregation dictionary for a dataframe.

    This function creates an aggregation dictionary for a dataframe,
    where each existing column in the dataframe is associated with the
    'first_non_nan' function. This dictionary can be used in aggregation
    operations, such as 'groupby.agg()'.

    Parameters:
    df (pandas.DataFrame): The dataframe for which to create the aggregation dictionary.

    Returns:
    dict: The aggregation dictionary.
    """

    # Create a list of all existing columns in the dataframe
    existing_columns = [col for col in df.columns]

    # Create an aggregation dictionary where each column is associated with the 'first_non_nan' function
    agg_dict = {col: first_non_nan for col in existing_columns}

    # Return the aggregation dictionary
    return agg_dict


if __name__ == "__main__":
    scite_abstracts = get_abstracts(n_abstracts=N_MOST_RELEVANT_PAPERS)
    logging.info(f"Total number of abstracts: {len(scite_abstracts)}")

    scite_abstracts = scite_abstracts.dropna(subset=["title", "abstract"])

    logging.info(
        f"Number of abstracts remaining after dropping NA titles and abstracts: {len(scite_abstracts)}"
    )
    scite_abstracts = scite_abstracts.drop_duplicates()
    logging.info(
        f"Number of abstracts remaining after dropping duplicates: {len(scite_abstracts)}"
    )

    # Create an aggregation dictionary for the dataframe
    agg_dict = create_agg_dict(scite_abstracts)

    # Group by 'doi' and aggregate
    scite_abstracts = (
        scite_abstracts.groupby("doi").agg(agg_dict).reset_index(drop=True)
    )
    logging.info(
        f"Number of abstracts remaining after grouping rows: {len(scite_abstracts)}"
    )

    scite_abstracts["abstract_clean"] = scite_abstracts["abstract"].apply(
        tc.clean_abstract
    )
    scite_abstracts = tc.clean_title_and_abstract(
        scite_abstracts, "abstract_clean", "title"
    )

    # Deduplicate again (some titles come up multiple times with different DOIs)
    scite_abstracts = scite_abstracts.drop_duplicates("title")
    logging.info(
        f"Number of abstracts remaining after deduplicating on titles: {len(scite_abstracts)}"
    )

    # Get rid of abstracts with very few citations, unless scite listed it as one of the core citations
    scite_abstracts: pd.DataFrame = scite_abstracts[
        (scite_abstracts["total_cites"] >= CITATION_THRESHOLD)
        | (scite_abstracts["category"] == "main")
    ]
    logging.info(
        f"Number of abstracts remaining after removing those with few citations: {len(scite_abstracts)}"
    )

    scite_embeddings = model.encode(
        scite_abstracts["title_abstract"].tolist(), show_progress_bar=True
    )

    vectors_as_list = [vec.tolist() for vec in scite_embeddings]

    # Add the embeddings as a column to the original dataframe
    scite_abstracts["embeddings"] = vectors_as_list

    scite_abstracts = scite_abstracts[
        [
            "title",
            "doi",
            "year",
            "abstract",
            "category",
            "supporting_cites",
            "contrasting_cites",
            "mentioning_cites",
            "total_cites",
            "abstract_clean",
            "title_abstract",
            "embeddings",
        ]
    ]

    # Save to parquet
    save_to_s3(S3_BUCKET, scite_abstracts, S3_KEY)
    # scite_abstracts.to_parquet(OUT_PATH)
