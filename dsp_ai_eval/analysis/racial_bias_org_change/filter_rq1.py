"""
Usage:
```
python dsp_ai_eval/analysis/racial_bias_org_change/filter_rq1.py --package-suffixes=.txt,.py,.yaml --datastore=s3 run
```
"""

import os

os.system(
    f"pip install --no-warn-conflicts --disable-pip-version-check -qr {os.path.dirname(os.path.realpath(__file__))}/metaflow_requirements.txt 1> /dev/null"  # nosec
)

from metaflow import FlowSpec, step, Parameter, S3, batch
import pandas as pd
from typing import List
import re


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


# def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


def unnest_works(df: pd.DataFrame) -> pd.DataFrame:
    # unnest some columns for desired data
    biblio = pd.json_normalize(df["biblio"])
    pmid = pd.json_normalize(df["ids"])[["pmid"]]
    primary_location = pd.json_normalize(df["primary_location"]).rename(
        columns={
            "source.display_name": "journal",
            "source.host_organization_name": "publisher",
            "source.issn_l": "issn_l",
        }
    )[
        ["journal", "publisher", "issn_l"]
    ]  # issn_l, source.display_name, source.host_organization_name
    primary_topics = pd.json_normalize(df["primary_topic"]).rename(
        columns=lambda x: f"primary_topic.{x}"
    )

    df = pd.concat([df, biblio, pmid, primary_location, primary_topics], axis=1)
    return df


def clean_works(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "doi",
            # "volume",
            # "issue",
            # "first_page",
            # "last_page",
            # "pmid",
            # "journal",
            # "publisher",
            # "issn_l",
        ]
    )
    print(f"Number of works remaining after dropping NA values on a subset: {len(df)}")

    df = df.drop_duplicates("doi")
    print(f"Number of works remaining after deduplicating on doi: {len(df)}")
    df = df.drop_duplicates("title")
    print(f"Number of works remaining after deduplicating on titles: {len(df)}")

    df["abstract_clean"] = df["abstract"].apply(clean_abstract)

    df = clean_title_and_abstract(
        df, "abstract_clean", "title"
    )  # adds title_abstract column

    # # count tokens in title_abstract
    # df["num_tokens"] = df["title_abstract"].apply(num_tokens_from_string)
    # print(
    #     "Number of documents more than 20000 tokens", len(df[df["num_tokens"] > 20000])
    # )

    # remove retracted papers
    df = df[df["is_retracted"] == False]
    print(f"Number of works remaining after filtering out retracted papers: {len(df)}")

    return df


def embed_works(sentences: List[str], model_name: str = "all-miniLM-L6-v2") -> List:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        sentences,
        show_progress_bar=True,
    )

    return embeddings.tolist()  # [vec.tolist() for vec in embeddings]


class RegexFilter(FlowSpec):
    batch_size = Parameter("batch_size", default=1000, help="Number of rows per batch")

    # production = Parameter("production", default=False, help="Run in production mode")
    # start_index = Parameter(
    #     "start_index", default=0, help="Starting index for processing"
    # )

    @batch(cpu=16, memory=64000)
    @step
    def start(self):

        import re

        self.rq_prefix = "racial_bias_org_change/rq1"
        self.path_filtered_data = "inputs/openalex/data/works_filtered.parquet"

        data_raw = pd.read_parquet(
            "s3://dsp-ai-eval/racial_bias_org_change/rq1/inputs/openalex/data/works_raw.parquet"
        )
        print(f"Number of records before filtering: {len(data_raw)}")
        self.data_raw = data_raw[data_raw["cited_by_count"] >= 3]
        print(
            f"Number of records after filtering out those with <3 citations: {len(self.data_raw)}"
        )

        # Define reusable regex components
        reducing_terms = r"reduc\w*|minimis\w*|mitigat\w*|remov\w*|eliminat\w*|solv\w*|improv\w*|fix\w*|counteract\w*|address\w*|tackl\w*|combat\w*"
        justice_terms = r"charg\w* decision\w*|justice system\w*|prosecut\w*|joint-enterprise|criminal justice|criminal justice system|court|law|legal|police|policing"
        learning_terms = r"learning|training|educat\w*|awareness|workshop|seminar|curriculum|program\w*|intervention"
        racial_bias_terms = r"racial bias|unconscious bias|implicit bias|racial framing|racism|racist|ethnic\w*|racial prejudice|racial discrimination|racial inequities|ethnic bias|racial disparities"
        workplace_terms = r"organisation|organization|workplace|company|business|institution|corporation|industry|sector"
        policy_terms = r"action plan|polic\w*|strateg\w*|framework|guidelines|intervention|initiative|roadmap"
        technology_terms = r"technolog\w*|digital|AI|artificial intelligence|machine learning|algorithmic bias|automated decision-making|predictive analytics"
        social_norms_terms = r"protocols|social norms|behavioral norms|behavioural norms|organizational culture|organisational culture|institutional policies|corporate practices|workplace norms"

        # Combine elements into regex patterns
        regex_patterns = {
            "reducing_bias_justice": rf"(?=.*\b({reducing_terms})\b)(?=.*\b({racial_bias_terms})\b)(?=.*\b({justice_terms})\b)",
            "bias_justice": rf"(?=.*\b({racial_bias_terms})\b)(?=.*\b({justice_terms})\b)",
            "learning_bias_workplace": rf"(?=.*\b({learning_terms})\b)(?=.*\b({racial_bias_terms})\b)(?=.*\b({workplace_terms})\b)",
            "learning_bias": rf"(?=.*\b({learning_terms})\b)(?=.*\b({racial_bias_terms})\b)",
            "changing_social_norms": rf"(?=.*\b({reducing_terms})\b)(?=.*\b({racial_bias_terms})\b)(?=.*\b({social_norms_terms})\b)",
            # "reducing_bias_AI": rf"(?=.*\b({reducing_terms})\b)(?=.*\b({racial_bias_terms})\b)(?=.*\b({justice_terms})\b)",
            "bias_workplace_policy": rf"(?=.*\b({racial_bias_terms})\b)(?=.*\b({workplace_terms})\b)(?=.*\b({policy_terms})\b)",
            "bias_policy_justice": rf"(?=.*\b({racial_bias_terms})\b)(?=.*\b({policy_terms})\b)(?=.*\b({justice_terms})\b)",
        }

        self.compiled_patterns = {
            category: re.compile(pattern, re.IGNORECASE)
            for category, pattern in regex_patterns.items()
        }

        # Split the data into batches
        print("Splitting data into batches...")
        # self.batches = list(enumerate(toolz.partition_all(self.batch_size, self.health_job_descriptions)))
        self.batches = list(
            enumerate(
                [
                    self.data_raw.iloc[i : i + self.batch_size]
                    for i in range(0, len(self.data_raw), self.batch_size)
                ]
            )
        )
        print(f"Total batches: {len(self.batches)}")

        self.next(self.process_data, foreach="batches")

    @batch(cpu=16, memory=64000)
    @step
    def process_data(self):

        def apply_compiled_regex(text):
            return {
                category: bool(pattern.search(text))
                for category, pattern in self.compiled_patterns.items()
            }

        batch_index, batch = self.input

        data_clean = batch.pipe(unnest_works).pipe(clean_works)

        data_clean["title_abstract"] = data_clean["title_abstract"].astype(str)

        # Apply regex filtering using .apply() but return multiple columns at once
        matches_df = (
            data_clean["title_abstract"].apply(apply_compiled_regex).apply(pd.Series)
        )

        # Merge results into original dataframe
        data_clean = pd.concat([data_clean, matches_df], axis=1)

        # Filter dataframe to keep rows matching at least one category
        self.df_filtered = data_clean[matches_df.any(axis=1)]
        self.filtered_path = f"s3://dsp-ai-eval/racial_bias_org_change/rq1/inputs/openalex/data/regexes/works_filtered_{batch_index}.parquet"
        self.df_filtered.to_parquet(self.filtered_path, index=False)

        self.next(self.join)

    @batch(cpu=8, memory=64000)
    @step
    def join(self, inputs):
        """Merge all batch results into a single output."""
        # Collect all batch results
        self.all_data = pd.concat(
            [pd.read_parquet(inp.filtered_path) for inp in inputs],
            ignore_index=True,
        )
        output_path = "s3://dsp-ai-eval/racial_bias_org_change/rq1/inputs/openalex/data/works_filtered.parquet"
        self.all_data.to_parquet(output_path, index=False)
        print(
            f"Saved works_filtered results with {len(self.all_data)} rows to {output_path}"
        )

        self.all_data["embeddings"] = embed_works(
            self.all_data["title_abstract"].tolist()
        )
        self.all_data.to_parquet(
            "s3://dsp-ai-eval/racial_bias_org_change/rq1/inputs/openalex/data/works_cleaned_w_embeddings.parquet",
            index=False,
        )
        print(f"Saved data with embeddings results with {len(self.all_data)} rows")

        self.next(self.end)

    @step
    def end(self):
        """End the flow."""
        print("Processing complete.")


if __name__ == "__main__":
    RegexFilter()
