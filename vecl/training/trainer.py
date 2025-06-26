import glob
import os
import shutil
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
        self.dataset_path = kwargs.pop('dataset_path', None)

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

    def _copy_embedding_files_if_needed(self):
        """Copy embedding files from dataset directory to run directory if they don't exist."""
        if not self.dataset_path:
            return

        run_output_dir = Path(self.output_path)
        dataset_dir = Path(self.dataset_path)

        # Files that might need copying from dataset to run directory
        embedding_files = ['speakers.pth', 'emotions.pth']

        for filename in embedding_files:
            run_file = run_output_dir / filename
            dataset_file = dataset_dir / filename

            if not run_file.exists() and dataset_file.exists():
                try:
                    shutil.copy2(dataset_file, run_file)
                    print(
                        f"    > 📁 Copied '{filename}' from dataset to run directory"
                    )
                except Exception as e:
                    print(f"    > ⚠️ Failed to copy '{filename}': {e}")

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

        # Copy embedding files if they don't exist in the run directory
        self._copy_embedding_files_if_needed()

        run_output_dir = Path(self.output_path)
        run_name = run_output_dir.name
        s3_run_path = f'{self.s3_prefix}/{run_name}'

        print(f'\n--- 🚀 Attempting S3 sync for: {s3_run_path} ---')
        try:
            files_to_upload = [
                'config.json',
                'language_ids.json',
                'speakers.pth',
                'emotions.pth',
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
