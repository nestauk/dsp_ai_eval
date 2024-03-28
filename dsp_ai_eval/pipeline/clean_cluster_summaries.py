import json
import pandas as pd
import re

from dsp_ai_eval import PROJECT_DIR

CLUSTER_SUMMARIES_INPATH = PROJECT_DIR / "outputs/data/cluster_summaries.json"
CLUSTER_SUMMARIES_OUTPATH = PROJECT_DIR / "outputs/data/cluster_summaries_cleaned.csv"


def to_snake_case(s):
    # Replace all non-word characters (everything except letters and numbers) with an underscore
    s = re.sub(r"\W+", "_", s)
    # Convert to lowercase
    s = s.lower()
    # Remove leading and trailing underscores
    s = s.strip("_")
    return s


def clean_column_names(df):
    """
    Converts all column names to snake case and strips leading or trailing punctuation.

    :param df: pandas DataFrame with any column names
    :return: pandas DataFrame with cleaned column names
    """
    new_columns = {col: to_snake_case(col) for col in df.columns}
    return df.rename(columns=new_columns)


if __name__ == "__main__":
    with open(CLUSTER_SUMMARIES_INPATH) as file:
        cluster_summaries = json.load(file)

    # Convert the nested dictionary into a list of dictionaries
    data = [v for _, v in cluster_summaries.items()]

    # Create a DataFrame
    df = pd.DataFrame(data).reset_index().rename(columns={"index": "topic"})

    df = clean_column_names(df)

    df = df.rename(
        columns={
            "docs": "representative_docs",
            "name": "topic_name",
            "description": "topic_description",
            "keywords": "topic_keywords",
        }
    )

    df[
        [
            "topic",
            "topic_name",
            "topic_description",
            "representative_docs",
            "topic_keywords",
        ]
    ].to_csv(CLUSTER_SUMMARIES_OUTPATH, index=False)
