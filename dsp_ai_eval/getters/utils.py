import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from decimal import Decimal
from fnmatch import fnmatch
import gzip
from io import BytesIO
import json
import numpy
import os
import pandas as pd
from pathlib import Path
import pickle
import shutil
import srsly
from typing import Optional, List, Dict
import yaml

from dsp_ai_eval import logger, S3_BUCKET

s3 = boto3.resource("s3")


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super(CustomJsonEncoder, self).default(obj)


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Loads a JSON Lines (JSONL) file into a list of dictionaries.

    Each line in the JSONL file should be a valid JSON object. Lines that are not valid JSON
    objects will be skipped, and an error message will be printed.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a JSON object
                    from a line in the JSONL file.

    Raises:
        ValueError: If a line in the file is not a valid JSON object.
    """
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                # Skip empty lines
                continue
            try:
                data.append(srsly.json_loads(line))
            except ValueError as e:  # srsly raises ValueError for JSON errors
                print(f"Error parsing JSON on line {line_number}: {e}")
                # Optionally, continue to next line or handle error differently
    return data


def load_s3_jsonl(
    bucket_name: str, s3_file_name: str, local_file: str
) -> Optional[List[Dict]]:
    """Downloads a jsonl from s3 and loads it into a list of dictionaries.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_file_name (str): The S3 key for the file to be downloaded.
        local_file (str): The local file path where the downloaded file will be saved.

    Returns:
        Optional[List[Dict]]: A list of dictionaries loaded from the JSONL file.
                              Returns None if the file could not be downloaded or parsed.

    Raises:
        FileNotFoundError: Raised if the specified file in the S3 bucket is not found OR the local path can't be found.
        NoCredentialsError: Raised if AWS credentials are not available.
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    output_file = None

    # Make sure that the directory where you want to store the file exists
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_name, local_file)
        logger.info(
            f"File {s3_file_name} downloaded from {bucket_name} to {local_file}"
        )

        output_file = load_jsonl(local_file)
    except FileNotFoundError:
        print(f"The file {s3_file_name} was not found in {bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"The file {s3_file_name} was not found in bucket {bucket_name}.")
        else:
            print(f"An error occurred: {e}")
    except NoCredentialsError:
        print("Credentials not available")

    return output_file


def upload_file_to_s3(local_file, bucket_name, s3_file_name):
    """
    Upload a file to an S3 bucket

    :param local_file: File to upload
    :param bucket_name: Bucket to upload to
    :param s3_file_name: S3 object name. If not specified then local_file is used
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Upload the file
        s3.upload_file(local_file, bucket_name, s3_file_name)
        print(f"File {local_file} uploaded to {bucket_name}/{s3_file_name}")
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
    except NoCredentialsError:
        print("Credentials not available")


def load_s3_data(bucket_name: str, file_name: str):
    """Load a file from an S3 location.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_name (str): Path to the file in the S3 bucket.

    Returns:
        Loaded data.
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    if fnmatch(file_name, "*.yml") or fnmatch(file_name, "*.yaml"):
        file = obj.get()["Body"].read().decode()
        return yaml.safe_load(file)
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.parquet"):
        return pd.read_parquet("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        with obj.get()["Body"] as f:
            return pickle.load(f)
    elif fnmatch(file_name, "*.npy"):
        # Load the .npy file directly from the S3 object
        buffer = BytesIO(obj.get()["Body"].read())  # Read the whole file into a buffer
        return numpy.load(buffer)  # Use numpy.load on the buffered data
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.parquet", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )


def save_to_s3(bucket_name: str, output_var, output_file_dir: str):
    """Saves a file to S3.

    Args:
        bucket_name (str): Bucket name.
        output_var (_type_): Output variable to save.
        output_file_dir (str): Path to save the file to.
    """

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.parquet"):
        output_var.to_parquet(
            "s3://" + bucket_name + "/" + output_file_dir, index=False
        )
    elif fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        obj.put(Body=pickle.dumps(output_var))
    elif fnmatch(output_file_dir, "*.gz"):
        obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
    elif fnmatch(output_file_dir, "*.txt"):
        obj.put(Body=output_var)
    elif fnmatch(output_file_dir, "*.npy"):
        buffer = BytesIO()
        numpy.save(buffer, output_var)
        buffer.seek(0)  # Move the cursor to the beginning of the buffer
        obj.put(Body=buffer.getvalue())
    elif fnmatch(output_file_dir, "*.json"):
        # Convert dictionary to JSON string
        json_string = json.dumps(output_var, cls=CustomJsonEncoder)
        # Writing the JSON data to S3
        obj.put(Body=json_string)
    else:
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def copy_folder_to_s3(local_path, bucket_name, destination):
    bucket = s3.Bucket(bucket_name)

    local_path = Path(local_path)  # Ensure path is a Path object
    for subdir, dirs, files in os.walk(str(local_path)):  # Convert Path to string here
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, "rb") as data:
                key = os.path.join(
                    destination, str(Path(full_path).relative_to(local_path))
                )  # Correctly handle Path objects
                bucket.put_object(Key=key, Body=data)


def download_directory_from_s3(bucket_name, s3_folder, local_dir):
    bucket = s3.Bucket(bucket_name)
    local_dir = Path(local_dir)

    # Remove existing directory and its contents if it exists
    if local_dir.exists():
        shutil.rmtree(local_dir)

    # Ensure the directory is created again
    local_dir.mkdir(parents=True, exist_ok=True)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        if obj.key.endswith("/"):
            continue  # skip directories
        target = local_dir / Path(obj.key).relative_to(s3_folder)
        target.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        bucket.download_file(obj.key, str(target))
