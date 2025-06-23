import argparse
import os
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm


def download_from_s3(bucket_name, s3_key, local_path):
    """
    Downloads a file from an S3 bucket, creating the destination folder if it
    does not exist.

    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Key of the file in the S3 bucket.
    :param local_path: Local path to save the downloaded file (can be a string
                       or a Path object).
    """
    # Ensure the destination directory exists.
    dest_path = Path(local_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client('s3')
    try:
        response = s3.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = int(response.get('ContentLength', 0))

        with tqdm(
            total=file_size, unit='B', unit_scale=True, desc=s3_key
        ) as pbar:
            s3.download_file(
                bucket_name,
                s3_key,
                str(dest_path),
                Callback=lambda bytes_transferred: pbar.update(
                    bytes_transferred
                ),
            )

        print(f'Successfully downloaded {s3_key} to {dest_path}')
    except Exception as e:
        print(f'Error downloading {s3_key}: {e}')
        raise


def extract_tar_file(tar_path, extract_path='.'):
    """
    Extracts a tar file (supports .tar.gz, .tar.bz2, and .tar).

    :param tar_path: Path to the tar file.
    :param extract_path: Directory to extract the files to.
    """
    try:
        print(f'Extracting {tar_path} to {extract_path}...')

        # Create extract directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Auto-detect compression format
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_path)
        print(f'Successfully extracted {tar_path} to {extract_path}')
    except Exception as e:
        print(f'Error extracting {tar_path}: {e}')
        raise


def download_s3_file(bucket_name: str, s3_key: str, local_path: Path) -> bool:
    """
    Downloads a file from S3 to a local path.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The key (path) of the file in the S3 bucket.
        local_path (Path): The local path to save the downloaded file.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    if local_path.exists():
        print(f'✔️ File already exists at {local_path}. Skipping download.')
        return True

    try:
        s3 = boto3.client('s3')
        print(f'⬇️ Downloading {s3_key} from bucket {bucket_name}...')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket_name, s3_key, str(local_path))
        print(f'✅ Download complete. File saved to {local_path}')
        return True
    except NoCredentialsError:
        print(
            '❌ S3 credentials not found. Please configure your AWS credentials.'
        )
        return False
    except Exception as e:
        print(f'❌ An error occurred during S3 download: {e}')
        return False


def main():
    """Main function to download and extract a file from S3."""
    parser = argparse.ArgumentParser(
        description='Download and extract a .tar.gz file from S3.'
    )
    parser.add_argument('--bucket-name', required=True, help='S3 bucket name.')
    parser.add_argument(
        '--s3-key', required=True, help='S3 key for the .tar.gz file.'
    )
    parser.add_argument(
        '--local-path',
        required=True,
        help='Local path to save the downloaded file.',
    )
    parser.add_argument(
        '--extract-path',
        default='.',
        help='Directory to extract the files to.',
    )

    args = parser.parse_args()

    # Ensure the directory for the local path exists
    os.makedirs(os.path.dirname(args.local_path), exist_ok=True)

    # Ensure the extraction directory exists
    os.makedirs(args.extract_path, exist_ok=True)

    download_from_s3(args.bucket_name, args.s3_key, args.local_path)
    extract_tar_file(args.local_path, args.extract_path)


if __name__ == '__main__':
    main()
