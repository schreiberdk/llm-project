import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def fetch_all_pdfs_from_s3(prefix: str, local_folder: str = "/tmp/pdf_data"):
    os.makedirs(local_folder, exist_ok=True)
    local_paths = []

    # List all objects under the prefix
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pdf"):  # only PDFs
            local_path = os.path.join(local_folder, os.path.basename(key))
            s3.download_file(BUCKET_NAME, key, local_path)
            local_paths.append(local_path)

    return local_paths
