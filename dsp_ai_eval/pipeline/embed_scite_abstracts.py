import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import PROJECT_DIR, logging, config
from dsp_ai_eval.utils import text_cleaning as tc

model = SentenceTransformer(config["embedding_model"])

SCITE_MAIN_REFERENCES = (
    PROJECT_DIR
    / "inputs/data/scite/How_does_technology_diffusion_impact_UK_growth_and_productivity_3F.csv"
)
SCITE_WIDER_REFERENCES1 = (
    PROJECT_DIR
    / "inputs/data/scite/technology diffusion impact on uk growth-2024-02-23.csv"
)
SCITE_WIDER_REFERENCES2 = (
    PROJECT_DIR
    / "inputs/data/scite/technology diffusion impact on uk productivity-2024-02-23.csv"
)

OUT_PATH = PROJECT_DIR / "inputs/data/embeddings/scite_embeddings.parquet"

CITATION_THRESHOLD = 5
N_MOST_RELEVANT_PAPERS = 1000


def read_scite_abstracts(filepath, category="main", n_abstracts=N_MOST_RELEVANT_PAPERS):
    df = pd.read_csv(filepath)

    logging.info(len(df))

    df["category"] = category

    # Scite by default sorts by relevance, so take only the N most relevant papers
    # (You can download up to 10,000! But that seems excessive)
    if len(df) > n_abstracts:
        df = df.head(n_abstracts)

    return df


def first_non_nan(series):
    return series.dropna().iloc[0] if not series.dropna().empty else np.nan


if __name__ == "__main__":
    scite_main_abstracts = read_scite_abstracts(SCITE_MAIN_REFERENCES, "main")
    scite_wider_abstracts1 = read_scite_abstracts(SCITE_WIDER_REFERENCES1, "wider")
    scite_wider_abstracts2 = read_scite_abstracts(SCITE_WIDER_REFERENCES2, "wider")

    scite_abstracts = pd.concat(
        [scite_main_abstracts, scite_wider_abstracts1, scite_wider_abstracts2]
    )
    logging.info(f"Total number of abstracts: {len(scite_abstracts)}")

    scite_abstracts = scite_abstracts.dropna(subset=["title", "abstract"])

    logging.info(
        f"Number of abstracts remaining after dropping NA titles and abstracts: {len(scite_abstracts)}"
    )
    scite_abstracts = scite_abstracts.drop_duplicates()
    logging.info(
        f"Number of abstracts remaining after dropping duplicates: {len(scite_abstracts)}"
    )

    agg_dict = {
        "date": first_non_nan,
        "title": first_non_nan,
        "doi": first_non_nan,
        "authors": first_non_nan,
        "journal": first_non_nan,
        "short_journal": first_non_nan,
        "volume": first_non_nan,
        "year": first_non_nan,
        "publisher": first_non_nan,
        "issue": first_non_nan,
        "page": first_non_nan,
        "abstract": first_non_nan,
        "category": first_non_nan,
        "pmid": first_non_nan,
        "issns": first_non_nan,
        "supporting_cites": first_non_nan,
        "contrasting_cites": first_non_nan,
        "mentioning_cites": first_non_nan,
        "total_cites": first_non_nan,
        "scite_report_link": first_non_nan,
    }

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
    scite_abstracts = scite_abstracts[
        (scite_abstracts["total_cites"] >= CITATION_THRESHOLD)
        | (scite_abstracts["category"] == "main")
    ]
    logging.info(
        f"Number of abstracts remaining after removing those with few citations: {len(scite_abstracts)}"
    )

    scite_embeddings = model.encode(
        scite_abstracts["title_abstract"].tolist(), show_progress_bar=True
    )

    vectors_as_list = [list(vec) for vec in scite_embeddings]

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
    scite_abstracts.to_parquet(OUT_PATH)
