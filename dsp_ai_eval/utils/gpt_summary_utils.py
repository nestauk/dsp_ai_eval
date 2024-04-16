import re
from typing import List


def extract_conclusion(answer: List[str]):
    last_item = answer[-1]
    if re.match(r"^[A-Za-z]", last_item):
        return last_item
    else:
        return None


def extract_theme_headings(text: str) -> List[str]:
    """
    Extracts and processes theme headings from a given text.

    This function searches the input text for patterns that start with a digit followed by a period,
    continuing up to the first colon in a line of text, extracts these headings, and processes them by removing
    any trailing punctuation (specifically, dots or colons at the end) and markdown bold syntax if present, then
    converts them to lowercase. The aim is to normalize the headings for further processing or analysis.

    Parameters:
    - text (str): A string of text that may contain theme headings.

    Returns:
    - List[str]: A list of the processed theme headings, with trailing punctuation removed, markdown bold syntax removed,
      leading and trailing whitespace stripped, and converted to lowercase.

    Example:
    - Input: "4. **Skills and Training**: Research examines the importance of skills development and training programs..."
    - Output: ['skills and training']
    """
    # Regular expression to match theme headings
    # This regex looks for a pattern that starts with a digit followed by a period, up to the first colon.
    theme_heading_regex = re.compile(r"\d+\.\s*(\*\*.*?\*\*|[^:*]*):")

    # Find all matches within the text
    matches = theme_heading_regex.findall(text)

    # Process each match
    headings = []
    for match in matches:
        # If the heading is surrounded by bold syntax "**", remove it
        heading = re.sub(r"^\*\*(.*?)\*\*$", r"\1", match)
        # Strip leading/trailing whitespace, remove trailing punctuation, and convert to lowercase
        heading = re.sub(r"[.:]+$", "", heading).strip().lower()
        headings.append(heading)

    return headings
