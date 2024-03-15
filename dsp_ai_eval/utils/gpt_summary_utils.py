import re
from typing import List


def extract_conclusion(answer: List[str]):
    last_item = answer[-1]
    if re.match(r"^[A-Za-z]", last_item):
        return last_item
    else:
        return None


# Regular expression to match theme headings
# This regex looks for lines that start with a number followed by a period and a space, capturing the rest of the line as the heading.
theme_heading_regex = re.compile(r"\*\*(.*?)\*\*")


def extract_theme_headings(text):
    # Directly find all matches within the text
    matches = theme_heading_regex.findall(text)
    # # Strip each match of leading/trailing whitespace and return the list
    # headings = [match.strip() for match in matches]
    # For each match, strip trailing punctuation, then strip leading/trailing whitespace, and convert to lowercase
    headings = [re.sub(r"[.:]+$", "", match).strip().lower() for match in matches]
    return headings
