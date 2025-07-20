import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional

from vecl.config import AppConfig
from vecl.data.storage import (
    Artifact,
    LocalBackend,
    S3StorageBackend,
    StorageBackend,
)

logger = logging.getLogger(__name__)


class DownloadManager:
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
        self.cache_dir = Path.home() / '.vecl_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: Dict[str, Artifact] = {}

    def register(self, artifact: Artifact):
        self.artifacts[artifact.name] = artifact

    def get(self, name: str) -> Optional[Path]:
        if name not in self.artifacts:
            raise ValueError(f'Artifact {name} not found')

        artifact = self.artifacts[name]

        if artifact.local_path.exists():
            logger.info(
                f'Artifact {name} already exists at {artifact.local_path}'
            )
            return artifact.local_path

        if self.storage_backend:
            self._download(artifact)

        elif artifact.required:
            raise ValueError(f'Artifact {name} not found and is required')

        else:
            logger.warning(f'Artifact {name} not found and is not required')
            return None

        return self.artifacts[name].local_path

    def _download(self, artifact: Artifact) -> bool:
        cache_file = self.cache_dir / f'{artifact.name}.download'

        if not cache_file.exists():
            logger.info(
                f'Downloading {artifact.name} from {self.backend.get_name()}'
            )
            downloaded = self.storage_backend.download(
                artifact.remote_path, cache_file
            )
            if not downloaded:
                raise RuntimeError(f'Failed to download {artifact.name}')

        if artifact.extract and artifact.extract_to:
            logger.info(f'Extracting {artifact.name}')
            self._extract(cache_file, artifact.extract_to)
        else:
            artifact.local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_file, artifact.local_path)

    def _extract(self, archive_path: Path, extract_to: Path):
        """Extract tar archive."""
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(extract_to)

    def list(self) -> Dict[str, dict]:
        """List all artifacts and their status."""
        return {
            name: {
                'exists': art.local_path.exists(),
                'required': art.required,
                'path': str(art.local_path),
            }
            for name, art in self.artifacts.items()
        }


def get_default_artifacts(config: AppConfig) -> Dict[str, Artifact]:
    return {
        'dataset': Artifact(
            name='dataset',
            remote_path=config.s3.data_key
            if config.s3
            else 'tts/cml-tts/processed_24k.tar.gz',
            local_path=config.paths.dataset_path / config.paths.metadata_file,
            extract=True,
            extract_to=config.paths.dataset_path,
            required=True,
        ),
        'yourtts_checkpoint': Artifact(
            name='yourtts_checkpoint',
            remote_path=config.s3.cml_tts_checkpoint_key
            if config.s3
            else 'tts/cml-tts/checkpoints_yourtts_cml_tts_dataset.tar.bz',
            local_path=config.paths.restore_path,
            extract=True,
            extract_to=config.paths.pretrained_checkpoint_dir.parent,
            required=True,
        ),
        'speaker_embeddings': Artifact(
            name='speaker_embeddings',
            remote_path=config.s3.speaker_embeddings_key
            if config.s3
            else 'tts/yourtts/embeddings/speaker_embeddings_patch.pth',
            local_path=config.paths.speaker_embeddings_file,
            required=False,
        ),
        'emotion_embeddings': Artifact(
            name='emotion_embeddings',
            remote_path=config.s3.emotion_embeddings_key
            if config.s3
            else 'tts/vecl-tts/embeddings/emotion_embeddings.pth',
            local_path=config.paths.emotion_embeddings_file,
            required=False,
        ),
    }


def create_download_manager(
    config: AppConfig,
    backend_type: Optional[str] = None,
    backend_config: Optional[dict] = None,
) -> DownloadManager:
    """Create a configured download manager."""
    backend = None
    if backend_type == 's3' and backend_config:
        backend = S3StorageBackend(backend_config['bucket_name'])
    elif backend_type == 'local' and backend_config:
        backend = LocalBackend(backend_config['base_path'])

    # Create manager and register artifacts
    dm = DownloadManager(storage_backend=backend)
    for artifact in get_default_artifacts(config).values():
        dm.register(artifact)

    return dm
