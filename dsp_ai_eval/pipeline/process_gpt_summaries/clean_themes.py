import numpy as np
import pandas as pd
import re
from typing import Union

from dsp_ai_eval import PROJECT_DIR, logging, config, S3_BUCKET
from dsp_ai_eval.utils import utils
from dsp_ai_eval.utils.gpt_summary_utils import extract_theme_headings
from dsp_ai_eval.getters.gpt import get_gpt_themes
from dsp_ai_eval.getters.utils import upload_file_to_s3

SEED = config["seed"]

GPT_THEMES_OUTPATH = "inputs/data/gpt/gpt_themes_repeats_cleaned.csv"

N_SAMPLES_PER_GROUP = 100

if __name__ == "__main__":
    answers_data = get_gpt_themes()

    answers_df = pd.DataFrame(answers_data)

    # When we first started collecting data, we did not record the system message.
    # Keep only those records where we know what the system message was.
    answers_filtered = answers_df[answers_df["system_message"].notna()]

    # Get a balanced sample (sometimes the process does not run completely, eg if your internet
    # connection is interrupted, so you may have more samples from one model/temp than another)
    min_group_size = answers_filtered.groupby(["gpt_model", "temperature"]).size().min()
    # Determine the sample size as the minimum between the desired size and the smallest group size
    final_sample_size = min(N_SAMPLES_PER_GROUP, min_group_size)
    answers_balanced = (
        answers_filtered.groupby(["gpt_model", "temperature"], group_keys=False)
        .apply(lambda x: x.sample(n=final_sample_size, random_state=SEED))
        .reset_index(drop=True)
    )

    # Currently the answers are stored as a list of strings, where each string is a line from
    # the GPT response. We want to explode this list so that each line is a separate row in the df.
    answers_long = answers_balanced.explode("answer")

    # Extract the heading from each line of each answer
    answers_long["heading"] = answers_long["answer"].apply(
        lambda x: extract_theme_headings(x)[0] if extract_theme_headings(x) else None
    )

    # # Define a regex pattern that matches a digit(s) followed by a period, any text until a colon
    # pattern = r"\d+\..*?:"
    # regex pattern to match digit(s) at line start followed by a period; or any asterisk anywhere in the line
    pattern = r"^\d+\.|[*]"

    answers_long["answer_cleaned"] = answers_long["answer"].apply(
        lambda x: re.sub(pattern, "", x)
    )

    answers_long.to_csv(PROJECT_DIR / GPT_THEMES_OUTPATH, index=False)

    # copy to s3
    upload_file_to_s3(PROJECT_DIR / GPT_THEMES_OUTPATH, S3_BUCKET, GPT_THEMES_OUTPATH)
