import json
from pathlib import Path


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e}")
                # Optionally, continue to next line or handle error differently
    return data
