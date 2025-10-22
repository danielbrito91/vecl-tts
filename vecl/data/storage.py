import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    name: str
    remote_path: str
    local_path: Path
    extract: bool = False
    extract_to: Optional[Path] = None
    required: bool = True


class StorageBackend(ABC):
    @abstractmethod
    def download(self, remote_path: str, local_path: Path) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.client('s3')

    def download(self, remote_path: str, local_path: Path) -> bool:
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name, Key=remote_path
            )
            file_size = int(response.get('ContentLength', 0))

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with tqdm(
                total=file_size, unit='B', unit_scale=True, desc=remote_path
            ) as pbar:
                self.client.download_file(
                    self.bucket_name,
                    remote_path,
                    str(local_path),
                    Callback=lambda bytes_transferred: pbar.update(
                        bytes_transferred
                    ),
                )

            return True
        except Exception as e:
            logger.error(f'Error downloading {remote_path}: {e}')
            return False

    def get_name(self) -> str:
        return f'S3:{self.bucket_name}'


class LocalBackend(StorageBackend):
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def download(self, remote_path: str, local_path: Path) -> bool:
        """Copy from local filesystem."""
        source = self.base_path / remote_path
        if not source.exists():
            logger.error(f'Source not found: {source}')
            return False

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            logger.info(f'Copied {source} -> {local_path}')
            return True
        except Exception as e:
            logger.error(f'Local copy failed: {e}')
            return False

    def get_name(self) -> str:
        return f'Local:{self.base_path}'
