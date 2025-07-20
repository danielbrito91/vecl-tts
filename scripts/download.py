import argparse
import logging
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from vecl.config import AppConfig
from vecl.data.downloader import create_download_manager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    config = AppConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )

    parser = argparse.ArgumentParser(description='Download VECL-TTS artifacts')
    parser.add_argument(
        '--artifacts',
        nargs='+',
        choices=['dataset', 'models', 'embeddings', 'all'],
        default=['all'],
        help='Which artifacts to download',
    )
    parser.add_argument(
        '--backend', choices=['s3'], default='s3', help='Storage backend'
    )
    parser.add_argument(
        '--s3-bucket',
        help='S3 bucket name (uses S3_BUCKET_NAME env var if not provided)',
    )
    parser.add_argument(
        '--local-mirror',
        type=Path,
        help='Path to local mirror (for local backend)',
    )
    parser.add_argument(
        '--list', action='store_true', help='List artifacts and exit'
    )

    args = parser.parse_args()
    load_dotenv()

    # Configure backend
    backend_type = None
    backend_config = None

    if args.backend == 's3':
        bucket = args.s3_bucket or os.getenv('S3_BUCKET_NAME')
        if bucket:
            backend_type = 's3'
            backend_config = {'bucket_name': bucket}
        else:
            logger.warning(
                'No S3 bucket specified. Only checking local files.'
            )
    elif args.backend == 'local' and args.local_mirror:
        backend_type = 'local'
        backend_config = {'base_path': args.local_mirror}

    # Create download manager
    dm = create_download_manager(config, backend_type, backend_config)

    # List and exit if requested
    if args.list:
        logger.info('\nArtifacts:')
        for name, info in dm.list().items():
            status = '✓' if info['exists'] else '✗'
            req = '*' if info['required'] else ' '
            logger.info(f'  {status} {req} {name:<20} {info["path"]}')
        logger.info('\n* = required')
        return

    # Determine what to download
    artifacts = []
    if 'all' in args.artifacts:
        artifacts = [
            'dataset',
            'yourtts_checkpoint',
            'speaker_embeddings',
            'emotion_embeddings',
        ]
    else:
        if 'dataset' in args.artifacts:
            artifacts.append('dataset')
        if 'models' in args.artifacts:
            artifacts.append('yourtts_checkpoint')
        if 'embeddings' in args.artifacts:
            artifacts.extend(['speaker_embeddings', 'emotion_embeddings'])

    # Download
    logger.info(f'\nDownloading: {", ".join(artifacts)}\n')
    for name in artifacts:
        try:
            _ = dm.get(name)
            logger.info(f'✓ {name}')
        except Exception as e:
            logger.error(f'✗ {name}: {e}')

    logger.info('\nDone!')


if __name__ == '__main__':
    main()
