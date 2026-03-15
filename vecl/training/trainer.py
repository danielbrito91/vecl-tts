import logging
from pathlib import Path
from typing import Optional

import boto3
from trainer import Trainer

from vecl.config import S3Config

logger = logging.getLogger(__name__)


class VeclTrainer(Trainer):
    """
    Custom Trainer that uploads model checkpoints and configs to an S3 bucket after saving locally.
    """

    def __init__(self, *args, s3_config: Optional[S3Config] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.s3_config = s3_config
        self.s3_client = None

        if s3_config:
            try:
                self.s3_client = boto3.client('s3')
                prefix = (
                    'vecl'
                    if 'vecl' in str(kwargs.get('output_path', ''))
                    else 'yourtts'
                )
                self.s3_prefix = f'tts/{prefix}/checkpoints'
                logger.info(
                    f'S3 uploads enabled: s3://{s3_config.bucket_name}/{self.s3_prefix}'
                )
            except Exception as e:
                logger.warning(f'S3 setup failed: {e}. Uploads disabled.')
                self.s3_client = None

    def save_checkpoint(self, *args, **kwargs):
        """Save checkpoint and optionally upload to S3."""
        checkpoint_path = super().save_checkpoint(*args, **kwargs)

        if checkpoint_path and self.s3_client and self.s3_config:
            self._upload_to_s3(checkpoint_path)

        return checkpoint_path

    def _upload_to_s3(self, checkpoint_path):
        """Upload checkpoint and config to S3."""
        try:
            run_name = Path(self.output_path).name
            files = [
                Path(checkpoint_path),
                Path(self.output_path) / 'config.json',
            ]

            for file in files:
                if file.exists():
                    s3_key = f'{self.s3_prefix}/{run_name}/{file.name}'
                    logger.info(f'Uploading {file.name} to S3...')
                    self.s3_client.upload_file(
                        str(file), self.s3_config.bucket_name, s3_key
                    )

        except Exception as e:
            logger.error(f'S3 upload failed: {e}')
