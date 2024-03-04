from dsp_ai_eval import PROJECT_DIR, logging

import argparse
from datetime import date
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo"
TEMP = 0.5
RQ = "How does technology diffusion impact UK growth and productivity?"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--n_samples",
        default=10,
        type=int,
        help="Number of times to repeatedly prompt GPT",
    )

    args = parser.parse_args()
    logging.info(args)

    themes_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Given a research question, you provide a summary of the current important themes in academic research on that topic.",
            ),
            ("user", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL, temperature=TEMP
    )

    output_parser = StrOutputParser()

    chain = themes_prompt | llm | output_parser

    answers = {}

    for i in range(args.n_samples):
        answer = chain.invoke({"input": RQ})
        answer = answer.split("\n")
        answers[i] = [line for line in answer if line != ""]

    with open(
        PROJECT_DIR
        / f"inputs/data/gpt/gpt_themes_repeats_{n_samples}_{date.today()}.json",
        "w",
    ) as outfile:
        json.dump(answers, outfile)
