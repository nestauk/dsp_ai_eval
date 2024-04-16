"""
Find out how long the output file is with:
```
wc -l inputs/data/gpt/gpt_themes_repeats.jsonl
```
"""

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

from dsp_ai_eval import PROJECT_DIR, logging, S3_BUCKET
from dsp_ai_eval.getters.utils import upload_file_to_s3

GPT_MODEL = "gpt-3.5-turbo"
TEMPS = [0, 0.25, 0.5, 1]
RQ = "How does technology diffusion impact UK growth and productivity?"
SYSTEM_MESSAGE = "You are a helpful research assistant. Given a research question, you provide a summary of the key topics in academic research on that topic."
N_SAMPLES = 50

# output
FILENAME = "inputs/data/gpt/gpt_themes_repeats.jsonl"
OUT_FILE = PROJECT_DIR / FILENAME

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
        help="Number of times to repeatedly prompt GPT per temp setting",
    )

    parser.add_argument(
        "--gpt_model",
        default=GPT_MODEL,
        type=str,
        help="GPT model",
    )

    parser.add_argument(
        "--production",
        default=False,
        type=bool,
        help="Run the script in production mode or test",
    )

    args = parser.parse_args()
    logging.info(args)

    if args.production == True:
        logging.info("running in production")
        outfile_path = OUT_FILE
        n_samples = args.n_samples
    else:
        logging.info("running in test mode")
        outfile_path = PROJECT_DIR / "inputs/data/gpt/gpt_themes_repeats_test.jsonl"
        n_samples = 2

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
        file_mode = "a" if outfile_path.exists() else "w"

        # Initialize the starting index
        start_index = 0

        if file_mode == "a":
            with open(outfile_path, "r") as infile:
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
                f"File {outfile_path} already exists and contains {start_index+1} samples. Appending instead of overwriting."
            )

        with open(outfile_path, file_mode) as outfile:
            for i in range(start_index, start_index + n_samples):
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
                    f"Model: {args.gpt_model}. Temp: {temp}. Sample {i+1}/{n_samples+start_index} done."
                )
                time.sleep(2)
            logging.info(
                f"You now have {n_samples+start_index} samples in {outfile_path}."
            )

    # copy the file to s3
    if args.production:
        upload_file_to_s3(outfile_path, S3_BUCKET, FILENAME)
