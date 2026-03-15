import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from vecl.config import AppConfig
from vecl.data.downloader import create_download_manager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    """Download VECL-TTS artifacts using configuration."""
    load_dotenv()
    config = AppConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )

    download_cfg = config.download

    # Configure backend
    backend_type = None
    backend_config = None

    if download_cfg.backend == 's3':
        bucket = download_cfg.s3_bucket or os.getenv('S3_BUCKET_NAME')
        if bucket:
            backend_type = 's3'
            backend_config = {'bucket_name': bucket}
        else:
            logger.warning(
                'No S3 bucket specified. Only checking local files.'
            )
    elif download_cfg.backend == 'local' and download_cfg.local_mirror:
        backend_type = 'local'
        backend_config = {'base_path': download_cfg.local_mirror}

    # Create download manager
    dm = create_download_manager(config, backend_type, backend_config)

    # List and exit if requested
    if download_cfg.list:
        logger.info('\nArtifacts:')
        for name, info in dm.list().items():
            status = '✓' if info['exists'] else '✗'
            req = '*' if info['required'] else ' '
            logger.info(f'  {status} {req} {name:<20} {info["path"]}')
        logger.info('\n* = required')
        return

    # Determine what to download
    artifacts_to_download = []
    if 'all' in download_cfg.artifacts:
        artifacts_to_download = [
            'dataset',
            'yourtts_checkpoint',
            'speaker_embeddings',
            'emotion_embeddings',
        ]
    else:
        if 'dataset' in download_cfg.artifacts:
            artifacts_to_download.append('dataset')
        if 'models' in download_cfg.artifacts:
            artifacts_to_download.append('yourtts_checkpoint')
        if 'embeddings' in download_cfg.artifacts:
            artifacts_to_download.extend([
                'speaker_embeddings',
                'emotion_embeddings',
            ])

    # Download
    logger.info(f'\nDownloading: {", ".join(artifacts_to_download)}\n')
    for name in artifacts_to_download:
        try:
            _ = dm.get(name)
            logger.info(f'✓ {name}')
        except Exception as e:
            logger.error(f'✗ {name}: {e}')

    logger.info('\nDone!')


if __name__ == '__main__':
    main()
