"""
Usage:
```
python dsp_ai_eval/analysis/racial_bias_org_change/extract_openalex.py --package-suffixes=.txt,.py,.yaml --datastore=s3 run
```
"""

import os

os.system(
    f"pip install --no-warn-conflicts --disable-pip-version-check -qr {os.path.dirname(os.path.realpath(__file__))}/metaflow_requirements.txt 1> /dev/null"  # nosec
)

from metaflow import FlowSpec, step, Parameter, S3, batch

import pandas as pd
from pathlib import Path
import pyalex
import re

# from dsp_ai_eval import config, S3_BUCKET, logging, PROJECT_DIR

S3_BUCKET = "dsp-ai-eval"
pyalex.config["email"] = "rosie.oxbury@nesta.org.uk"
MIN_CITES = ">0"
# RAW_DATA_PATH = "inputs/openalex/data/works_raw.parquet"


def format_string(input_string: str) -> str:
    def replace_spaces(match):
        return match.group(0).replace(" ", "+")

    formatted_string = re.sub(r".*?", replace_spaces, input_string)
    return formatted_string


def get_works(
    search="[Learning+OR+training]+AND+[anti-racist]+AND+[organisation+OR+work+OR+workplace]",
    citation_filter=None,
):

    if citation_filter is not None:
        query = pyalex.Works().search(search).filter(cited_by_count=citation_filter)
    else:
        query = pyalex.Works().search(search)

    results = []
    for page in query.paginate(per_page=200, n_max=80000):
        results.extend(page)

    for page in results:
        page["abstract"] = page["abstract"]

    if len(results) > 0:
        df = (
            pd.DataFrame(results)
            .dropna(subset=["title", "abstract"])
            .drop(columns=["abstract_inverted_index"])
        )
    else:
        df = pd.DataFrame()

    return df


class OpenAlexSearchFlow(FlowSpec):

    min_cites = Parameter(
        "min_cites", default=MIN_CITES, help="Minimum citation filter"
    )

    @step
    def start(self):
        """Load search terms and prepare formatted queries."""

        self.s3_bucket = S3_BUCKET

        searches_list = pd.read_csv(
            "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/rq1_rq2_search_terms - rq2_search_terms v2.csv"
        )["search"]

        search_terms = [format_string(x) for x in searches_list]
        self.searches = list(enumerate(search_terms))
        self.next(self.fetch_works, foreach="searches")

    @batch(cpu=16, memory=64000)
    @step
    def fetch_works(self):

        search_index, search_term = self.input

        temp_df = get_works(search=search_term, citation_filter=self.min_cites)
        print(f"Search: {search_term}, Number of results: {len(temp_df)}")

        temp_df = temp_df.assign(search_term=search_term)

        self.search_results_path = f"s3://{self.s3_bucket}/racial_bias_org_change/rq2/inputs/openalex/data/raw_searches/works_raw_{search_index}.parquet"
        temp_df.to_parquet(self.search_results_path)

        self.next(self.join)

    @batch(cpu=8, memory=64000)
    @step
    def join(self, inputs):
        """Merge all batch results into a single output."""
        # Collect all batch results
        all_data = pd.concat(
            [pd.read_parquet(inp.search_results_path) for inp in inputs],
            ignore_index=True,
        )

        all_data.to_parquet(
            "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_raw.parquet"
        )

        self.next(self.end)

    @step
    def end(self):
        """Flow ends here."""
        print("Metaflow pipeline completed successfully.")


if __name__ == "__main__":
    OpenAlexSearchFlow()


# if __name__ == "__main__":
#     searches = pd.read_csv(
#         PROJECT_DIR / "dsp_ai_eval/analysis/racial_bias_org_change/rq1_rq2_search_terms - rq2_search_terms.csv"
#     )["search"]

#     input_searches_formatted = [format_string(x) for x in searches]
#     additional_terms_formatted = [format_string(x) for x in additional_terms]
#     combined_searches = [
#         f"{search}+AND+{term}"
#         for search in input_searches_formatted
#         for term in additional_terms_formatted
#     ]  # not currently used

#     outputs = {}

#     total_len = 0

#     chunk = 0
#     for search in input_searches_formatted:
#         temp_df = get_works(
#             search=search, citation_filter=config["oa_abstracts_pipeline"]["min_cites"]
#         )
#         logging.info(f"Search: {search}, Number of results: {len(temp_df)}")
#         total_len += len(temp_df)
#         temp_df.to_parquet(
#         f"s3://{S3_BUCKET}/racial_bias_org_change/rq2/inputs/openalex/data/raw_searches/works_raw_{chunk}.parquet")
#         chunk += 1
#         outputs[search] = temp_df

#     logging.info(f"Total results: {total_len}")
#     save_to_s3(S3_BUCKET, outputs, f"s3://{S3_BUCKET}/racial_bias_org_change/rq2/inputs/openalex/data/raw_searches/works_raw.json")


#     combined_df = pd.concat(
#         [df.assign(search_term=search_term) for search_term, df in outputs.items()],
#         ignore_index=True,
#     )

#     combined_df.to_parquet(
#         f"s3://{S3_BUCKET}/racial_bias_org_change/rq1/{config['oa_abstracts_pipeline']['path_raw_data']}"
#     )
