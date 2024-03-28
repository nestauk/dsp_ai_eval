"""
Find out how long the output file is with:
```
wc -l inputs/data/gpt/gpt_themes_repeats.jsonl
```
"""

from dsp_ai_eval import PROJECT_DIR, logging

import argparse
from datetime import date, datetime
from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
import time

GPT_MODEL = "gpt-4"
TEMPS = [0, 0.25, 0.5, 1]
RQ = "How does technology diffusion impact UK growth and productivity?"
SYSTEM_MESSAGE = "You are a helpful research assistant. Given a research question, you provide a summary of the key topics in academic research on that topic."
N_SAMPLES = 20

# output
OUT_FILE = PROJECT_DIR / f"inputs/data/gpt/gpt_themes_repeats.jsonl"

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--n_samples",
        default=N_SAMPLES,
        type=int,
        help="Number of times to repeatedly prompt GPT",
    )

    parser.add_argument(
        "--gpt_model",
        default=GPT_MODEL,
        type=str,
        help="GPT model",
    )

    args = parser.parse_args()
    logging.info(args)

    themes_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_MESSAGE,
            ),
            ("user", "{input}"),
        ]
    )

    for temp in TEMPS:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name=args.gpt_model, temperature=temp
        )

        output_parser = StrOutputParser()

        chain = themes_prompt | llm | output_parser

        # Determine the file's opening mode ('a' for append, 'w' for write)
        file_mode = "a" if Path(OUT_FILE).exists() else "w"

        # Initialize the starting index
        start_index = 0

        if file_mode == "a":
            with open(OUT_FILE, "r") as infile:
                # Read the file to find the highest index
                try:
                    # Extract the keys (indices) and find the maximum
                    start_index = (
                        max(int(next(iter(json.loads(line)))) for line in infile) + 1
                    )
                except ValueError:
                    # If the file is empty or keys are not integers, start_index remains 0
                    start_index = 0
            logging.info(
                f"File {OUT_FILE} already exists and contains {start_index+1} samples. Appending instead of overwriting."
            )

        with open(OUT_FILE, file_mode) as outfile:
            for i in range(start_index, start_index + args.n_samples):
                answer = chain.invoke({"input": RQ})
                answer = answer.split("\n")
                answer_clean = [line for line in answer if line != ""]
                datetime_str = datetime.now().isoformat()
                json_record = json.dumps(
                    {
                        "answer": answer_clean,
                        "datetime": datetime_str,
                        "gpt_model": args.gpt_model,
                        "temperature": temp,
                        "research_question": RQ,
                        "system_message": SYSTEM_MESSAGE,
                    }
                )
                outfile.write(json_record + "\n")
                logging.info(
                    f"Model: {args.gpt_model}. Temp: {temp}. Sample {i+1}/{args.n_samples+start_index} done."
                )
                time.sleep(2)
            logging.info(
                f"You now have {args.n_samples+start_index} samples in {OUT_FILE}."
            )
