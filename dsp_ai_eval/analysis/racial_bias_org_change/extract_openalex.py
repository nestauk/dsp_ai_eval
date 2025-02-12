import pandas as pd
import pyalex
import re

from dsp_ai_eval import config, S3_BUCKET, logging, PROJECT_DIR

pyalex.config["email"] = config["openalex_user"]


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
    for page in query.paginate(per_page=200, n_max=100000):
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


additional_terms = [
    "[UK OR United Kingdom OR Canada OR Australia OR US OR United States OR EU OR European Union OR Commonwealth]",
    "[the Crown Office and Procurator Fiscal Service OR the Public Prosecution Service OR the Public Prosecution Service of Canada OR the Office of the Commonwealth Director of Public Prosecutions OR the National Prosecuting Authority]",
    "[effective OR effectiveness OR evaluation OR RCT OR randomised controlled trial OR meta-analysis]",
]

if __name__ == "__main__":
    searches = pd.read_csv(
        PROJECT_DIR / "dsp_ai_eval/analysis/racial_bias_org_change/search_terms.csv"
    )["search"]

    input_searches_formatted = [format_string(x) for x in searches]
    additional_terms_formatted = [format_string(x) for x in additional_terms]
    combined_searches = [
        f"{search}+AND+{term}"
        for search in input_searches_formatted
        for term in additional_terms_formatted
    ]  # not currently used

    outputs = {}

    total_len = 0

    for search in input_searches_formatted:
        temp_df = get_works(
            search=search, citation_filter=config["oa_abstracts_pipeline"]["min_cites"]
        )
        logging.info(f"Search: {search}, Number of results: {len(temp_df)}")
        total_len += len(temp_df)
        outputs[search] = temp_df

    logging.info(f"Total results: {total_len}")

    combined_df = pd.concat(
        [df.assign(search_term=search_term) for search_term, df in outputs.items()],
        ignore_index=True,
    )

    combined_df.to_parquet(
        f"s3://{S3_BUCKET}/{config['rq_prefix']}/{config['oa_abstracts_pipeline']['path_raw_data']}"
    )
