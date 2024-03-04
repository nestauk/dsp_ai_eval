import pandas as pd
from sentence_transformers import SentenceTransformer

from dsp_ai_eval import PROJECT_DIR, logging
from dsp_ai_eval.utils import text_cleaning as tc

model = SentenceTransformer("all-miniLM-L6-v2")

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


def deduplicate_scite_dois(filepath, category="main", dois=[]):
    df = pd.read_csv(filepath)
    if dois == []:
        dois = df["doi"].unique().tolist()
    else:
        logging.info(len(df))
        df = df[~df["doi"].isin(dois)]
        logging.info(len(df))
        dois = dois + df["doi"].unique().tolist()
    df["category"] = category
    logging.info(f"Number of unique DOIs: {len(dois)}")
    return df[["title", "abstract", "year", "category"]], dois


if __name__ == "__main__":

    scite_main_abstracts, dois = deduplicate_scite_dois(SCITE_MAIN_REFERENCES, "main")
    scite_wider_abstracts1, dois = deduplicate_scite_dois(
        SCITE_WIDER_REFERENCES1, "wider", dois
    )
    scite_wider_abstracts2, dois = deduplicate_scite_dois(
        SCITE_WIDER_REFERENCES2, "wider", dois
    )

    scite_abstracts = pd.concat(
        [scite_main_abstracts, scite_wider_abstracts1, scite_wider_abstracts2]
    )
    scite_abstracts = scite_abstracts.dropna(subset=["title", "abstract"])
    scite_abstracts["abstract_clean"] = scite_abstracts["abstract"].apply(
        tc.clean_abstract
    )
    scite_abstracts = tc.clean_title_and_abstract(
        scite_abstracts, "abstract_clean", "title"
    )

    scite_embeddings = model.encode(
        scite_abstracts["title_abstract"].tolist(), show_progress_bar=True
    )

    vectors_as_list = [list(vec) for vec in scite_embeddings]

    # Add the embeddings as a column to the original dataframe
    scite_abstracts["embeddings"] = vectors_as_list

    # Save to parquet
    scite_abstracts.to_parquet(OUT_PATH)
