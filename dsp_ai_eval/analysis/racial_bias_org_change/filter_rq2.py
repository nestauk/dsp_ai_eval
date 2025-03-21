"""
Usage:
```
python dsp_ai_eval/analysis/racial_bias_org_change/filter_rq2.py --package-suffixes=.txt,.py,.yaml --datastore=s3 run
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

    production = Parameter("production", default=False, help="Run in production mode")
    # start_index = Parameter(
    #     "start_index", default=0, help="Starting index for processing"
    # )

    @batch(cpu=32, memory=128000)
    @step
    def start(self):

        import re

        self.rq_prefix = "racial_bias_org_change/rq2"
        self.path_filtered_data = "inputs/openalex/data/works_filtered.parquet"

        data_raw = pd.read_parquet(
            "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_raw.parquet"
        )
        print(f"Number of records before filtering: {len(data_raw)}")
        data_raw = data_raw[data_raw["cited_by_count"] >= 3]
        print(
            f"Number of records after filtering out those with <3 citations: {len(data_raw)}"
        )
        self.data_raw = data_raw[data_raw["relevance_score"] >= 5]
        print(
            f"Number of records after filtering out those with relevance scores <5: {len(self.data_raw)}"
        )

        self.data_raw = self.data_raw.dropna(
            subset=[
                "doi",
            ]
        )
        print(
            f"Number of works remaining after dropping NA values on a subset: {len(self.data_raw)}"
        )

        self.data_raw = self.data_raw.drop_duplicates("doi")
        print(
            f"Number of works remaining after deduplicating on doi: {len(self.data_raw)}"
        )
        self.data_raw = self.data_raw.drop_duplicates("title")
        print(
            f"Number of works remaining after deduplicating on titles: {len(self.data_raw)}"
        )

        if not self.production:
            self.data_raw = self.data_raw.sample(1000)

        # Define reusable regex components
        organisational_change = r"(?:organi[sz]ation\w*\s+)?(?:cultur\w*\s+)?(?:change|transformation|shift|reform\w*)"
        dei = r"equity|equality|diversity|inclusion|inclusivity|anti-racism"
        racial_bias = r"racial bias|unconscious bias|implicit bias|racial framing|racism|racial prejudice|racial discrimination"
        leadership_management = r"leader\w*|management|communication\w*|system\w*|structure\w*|process\w*|attitude\w*|behaviou?r\w*|value\w*"
        barriers_enablers = r"barrier|enable\w*|block\w*|hinder|help"
        accountability_incentives = r"accountab\w*|incentiv\w*"

        # Combine elements into regex patterns
        regex_patterns = {
            # "org_change": rf"\b({organisational_change})\b",
            "org_culture_racial_bias": rf"(?=.*\b({organisational_change})\b)(?=.*\b({racial_bias})\b)",
            "org_culture_equity": rf"(?=.*\b({organisational_change})\b)(?=.*\b({dei})\b)",
            "org_culture_leadership": rf"(?=.*\b({organisational_change})\b)(?=.*\b({leadership_management})\b)",
            "barriers_enablers_org_culture": rf"(?=.*\b({barriers_enablers})\b)(?=.*\b({organisational_change})\b)",
            "accountability_incentives_org_culture": rf"(?=.*\b({accountability_incentives})\b)(?=.*\b({organisational_change})\b)",
            "accountability_equity_org_culture": rf"(?=.*\b({accountability_incentives})\b)(?=.*\b({dei})\b)",
            "increasing_cultural_competence": rf"(?=.*\b(increas\w*\s+cultur\w*\s+competence)\b)",
            "supporting_managing_staff_org_culture": rf"(?=.*\b(support\w*|manag\w*)\b)(?=.*\b(staff|stakeholders|team)\b)(?=.*\b({organisational_change})\b)",
        }

        for category, pattern in regex_patterns.items():
            try:
                print(f"Compiling {category}: {pattern}")
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                print(f"Regex compilation error in '{category}': {e}")

        self.compiled_patterns = {
            category: re.compile(pattern, re.IGNORECASE)
            for category, pattern in regex_patterns.items()
        }

        # Split the data into batches
        print("Splitting data into batches...")
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

    @batch(cpu=32, memory=128000)
    @step
    def process_data(self):

        def apply_compiled_regex(text):
            return {
                category: bool(pattern.search(text))
                for category, pattern in self.compiled_patterns.items()
            }

        batch_index, batch = self.input

        # if not self.production:
        #     batch = batch.sample(10)

        data_clean = batch.pipe(unnest_works).pipe(clean_works)

        data_clean["title_abstract"] = data_clean["title_abstract"].astype(str)

        # Apply regex filtering using .apply() but return multiple columns at once
        matches_df = (
            data_clean["title_abstract"].apply(apply_compiled_regex).apply(pd.Series)
        )

        # Merge results into original dataframe
        data_clean = pd.concat([data_clean, matches_df], axis=1)

        # Filter dataframe to keep rows matching at least one category
        df_filtered = data_clean[matches_df.any(axis=1)]
        print(f"Final number of records for batch {batch_index}: {len(df_filtered)}")

        if self.production:
            self.filtered_path = f"s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/regexes/works_filtered_{batch_index}.parquet"
        else:
            self.filtered_path = f"s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/regexes/works_filtered_{batch_index}_test.parquet"
        df_filtered.to_parquet(self.filtered_path, index=False)

        self.next(self.join)

    @batch(cpu=32, memory=128000)
    @step
    def join(self, inputs):
        """Merge all batch results into a single output."""
        # Collect all batch results
        self.all_data = pd.concat(
            [pd.read_parquet(inp.filtered_path) for inp in inputs],
            ignore_index=True,
        )

        if self.production:
            output_path = "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_filtered_rx.parquet"
        else:
            output_path = "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_filtered_rx_test.parquet"

        self.all_data.to_parquet(output_path, index=False)
        print(
            f"Saved works_filtered results with {len(self.all_data)} rows to {output_path}"
        )

        self.all_data["embeddings"] = embed_works(
            self.all_data["title_abstract"].tolist()
        )

        if self.production:
            self.all_data.to_parquet(
                "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_cleaned_w_embeddings.parquet",
                index=False,
            )
        else:
            self.all_data.to_parquet(
                "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_cleaned_w_embeddings_test.parquet",
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
