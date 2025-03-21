"""
Performs multiple OpenAlex searches using search terms defined in an input .csv file.
Concatenates the results of all searches into a single output file.

Note that no deduplication is done in this script.

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
from typing import Optional

S3_BUCKET = "dsp-ai-eval"
pyalex.config["email"] = "rosie.oxbury@nesta.org.uk"
MIN_CITES = ">0"
SEARCH_FILE = "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/rq1_rq2_search_terms - rq2_search_terms v2.csv"


def format_string(input_string: str) -> str:
    """Format search terms into strings that can be pasted into the API call"""

    def replace_spaces(match):
        return match.group(0).replace(" ", "+")

    formatted_string = re.sub(r".*?", replace_spaces, input_string)
    return formatted_string


def get_works(
    search: str = "[Learning+OR+training]+AND+[anti-racist]+AND+[organisation+OR+work+OR+workplace]",
    citation_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query the OpenAlex Works API using a search string and optional citation filter,
    returning a DataFrame of relevant results with titles and abstracts.

    Args:
        search (str): The query string for searching works. Defaults to a query about
                      anti-racist learning or training in organisational/workplace contexts.
        citation_filter (Optional[str]): A string expression used to filter results based on
                      citation count (e.g., '>100', '10', '<=50'). If None, no citation filter is applied.

    Returns:
        pd.DataFrame: A DataFrame containing the resulting works, filtered to include only
                      rows with both 'title' and 'abstract'.
    """
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

        searches_list = pd.read_csv(SEARCH_FILE)["search"]

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
