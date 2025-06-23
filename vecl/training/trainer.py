import glob
import os
from pathlib import Path

import boto3
from trainer import Trainer


class UnifiedTrainer(Trainer):
    """
    Custom Trainer that uploads model checkpoints and configs to an S3 bucket after saving locally.
    """

    def __init__(self, *args, **kwargs):
        self.s3_bucket = kwargs.pop('s3_bucket', None)
        self.s3_prefix = kwargs.pop('s3_prefix', 'checkpoints')

        if self.s3_bucket and boto3:
            print(
                f'✅ S3 Uploader is active. Checkpoints will be sent to s3://{self.s3_bucket}/{self.s3_prefix}'
            )
            self.s3_client = boto3.client('s3')
        else:
            print(
                'ℹ️  S3 Uploader is disabled. No bucket name provided or boto3 is not installed.'
            )
            self.s3_client = None
        super().__init__(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        checkpoint_path = super().save_checkpoint(*args, **kwargs)
        if not self.s3_client:
            if checkpoint_path:
                print('   > ℹ️ S3 upload skipped.')
            return checkpoint_path

        if not checkpoint_path:
            print('   > 🕵️ Checkpoint path not returned. Searching manually...')
            checkpoint_files = glob.glob(
                f'{self.output_path}/checkpoint_*.pth'
            )
            if not checkpoint_files:
                print('   > ❌ No checkpoint files found for upload.')
                return None
            checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
            print(f'   > ✅ Latest checkpoint found: {checkpoint_path}')

        run_output_dir = Path(self.output_path)
        run_name = run_output_dir.name
        s3_run_path = f'{self.s3_prefix}/{run_name}'

        print(f'\n--- 🚀 Attempting S3 sync for: {s3_run_path} ---')
        try:
            files_to_upload = [
                'config.json',
                'language_ids.json',
                'speakers.pth',
                'emotion_embeddings.pth',
                Path(checkpoint_path).name,
            ]
            print(f'    > Files to upload: {files_to_upload}')
            for filename in files_to_upload:
                local_file = run_output_dir / filename
                if local_file.exists():
                    s3_key = f'{s3_run_path}/{filename}'
                    print(f"    > Uploading '{filename}' to '{s3_key}'...")
                    self.s3_client.upload_file(
                        str(local_file), self.s3_bucket, s3_key
                    )
                    print(f"    > Upload of '{filename}' complete.")
                else:
                    print(
                        f"    > WARNING: Local file '{filename}' not found. Skipping upload."
                    )
            print('    ✓ S3 sync finished successfully.')
        except Exception as e:
            print(f'    ❌ CRITICAL ERROR during S3 upload: {e}')

        print(
            '------------------------------------------------------------------'
        )
        return checkpoint_path
