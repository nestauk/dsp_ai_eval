import re
import pandas as pd


def clean_abstract(text):
    # remove \r, \n
    rep = r"|".join((r"\r", r"\n"))
    output = re.sub(rep, "", text)

    # remove duplicated punctuation
    output = re.sub(
        r"([!()\-{};:,<>./?@#$%\^&*_~]){2,}", lambda x: x.group()[0], output
    )

    # remove extra space
    output = re.sub(r"\s+", " ", output).strip()

    # Remove the word 'abstract' at the start
    abstract_regex = r"^[Aa]bstract[,.!?;:\-]?"
    output = re.sub(abstract_regex, "", output)

    # Remove html tags
    tags_regex = r"<[^>]+>"
    output = re.sub(tags_regex, "", output)

    return output


def clean_title_and_abstract(
    df: pd.DataFrame, abstract_col="abstract_clean", title_col="title"
):
    df["title_abstract"] = df.apply(
        lambda row: row[title_col]
        + (". " if not row[title_col].endswith(".") else " ")
        + row[abstract_col],
        axis=1,
    )
    return df
