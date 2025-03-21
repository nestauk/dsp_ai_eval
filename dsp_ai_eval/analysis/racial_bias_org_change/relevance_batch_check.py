import pandas as pd
import plac
from pathlib import Path
from discovery_utils.utils.llm.batch_check import LLMProcessor

# from dsp_ai_eval import PROJECT_DIR
PROJECT_DIR = Path(__file__).resolve().parents[3]
OUT_FILE = (
    PROJECT_DIR
    / "dsp_ai_eval/analysis/racial_bias_org_change/data/relevance_batch_check.jsonl"
)
FINAL_OUT_PATH = (
    PROJECT_DIR
    / "dsp_ai_eval/analysis/racial_bias_org_change/data/relevance_batch_check_eval.csv"
)


def main(production=False):

    # print(PROJECT_DIR)
    # print(OUT_FILE)

    data_clean = pd.read_parquet(
        "s3://dsp-ai-eval/racial_bias_org_change/rq2/inputs/openalex/data/works_cleaned_w_embeddings.parquet"
    )

    data_clean = data_clean[data_clean["cited_by_count"] >= 5]
    print(
        f"Number of abstracts after removing those with <5 citations: {len(data_clean)}"
    )

    if not production:
        data_clean = data_clean.sample(1000)

    text_data = data_clean.set_index("id")["title_abstract"].to_dict()

    system_message = """
    You have five tasks.
    1. Determine whether this text is relevant to the research question 'What does current literature and practice tell us about the most effective way to enable organisational cultural change?'
    2. Determine whether the research described in this text involves empirical evidence (including meta-analyses) such as an observational study or control trial.
    3. If the answer to the second question is 'yes', please identify the evidence type mentioned in the text.
    4. Does the text mention changing organizational culture in the context of diversity and inclusion? For example, DEI initiatives, reducing racism, reducing unconscious bias and so on.
    5. Does the text mention organizational change or culture change in a justice-related context such as criminal courts, Crown Office, Public Prosecution Service and similar organisations?
    """

    fields = [
        {
            "name": "is_relevant",
            "type": "str",
            "description": "A one-word answer: 'yes' or 'no'.",
        },
        {
            "name": "empirical",
            "type": "str",
            "description": "A one-word answer: 'yes' or 'no'.",
        },
        {
            "name": "evidence_type",
            "type": "str",
            "description": "What evidence type is mentioned in the text?",
        },
        {
            "name": "dei",
            "type": "str",
            "description": "A one-word answer: 'yes' or 'no'.",
        },
        {
            "name": "justice",
            "type": "str",
            "description": "A one-word answer: 'yes' or 'no'.",
        },
    ]

    processor = LLMProcessor(
        output_path=OUT_FILE,  # path to the output file
        system_message=system_message,  # system message
        session_name="test_abstract_relevance",  # used for tracking usage on LangFuse
        output_fields=fields,  # your output data fields
    )

    processor.run(text_data)

    output = pd.read_json(OUT_FILE, lines=True)

    out_data = pd.merge(data_clean, output, on="id")
    out_data[
        [
            "id",
            "title_abstract",
            "is_relevant",
            "empirical",
            "evidence_type",
            "dei",
            "justice",
        ]
    ].to_csv(FINAL_OUT_PATH)


if __name__ == "__main__":
    plac.call(main)
