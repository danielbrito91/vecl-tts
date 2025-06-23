import tarfile
import tempfile
from pathlib import Path

import boto3


def download_and_extract(bucket_name, s3_key, extract_to, cleanup=True):
    """
    Download and extract a tar.gz file from S3.

    Args:
        bucket_name (str): S3 bucket name
        s3_key (str): S3 object key (path to tar.gz file)
        extract_to (str): Local directory to extract to
        cleanup (bool): Remove downloaded tar.gz file after extraction

    Returns:
        str: Path to extracted directory
    """
    # Create S3 client (uses SageMaker IAM role automatically)
    s3 = boto3.client('s3')

    # Create extraction directory
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Create temporary file for download
    temp_file = Path(tempfile.gettempdir()) / Path(s3_key).name

    try:
        print(f'Downloading s3://{bucket_name}/{s3_key}...')
        s3.download_file(bucket_name, s3_key, str(temp_file))
        print(f'Downloaded to {temp_file}')

        print(f'Extracting to {extract_path}...')
        with tarfile.open(temp_file, 'r:gz') as tar:
            tar.extractall(extract_path)
        print('Extraction complete!')

        return str(extract_path)

    finally:
        # Clean up temp file
        if cleanup and temp_file.exists():
            temp_file.unlink()
            print(f'Cleaned up {temp_file}')


def list_s3_files(bucket_name, prefix=''):
    """List files in S3 bucket with optional prefix."""
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    return []
