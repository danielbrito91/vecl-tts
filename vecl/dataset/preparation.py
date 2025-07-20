import shutil
from pathlib import Path

import pandas as pd
from TTS.config.shared_configs import BaseDatasetConfig

from vecl.config import AppConfig
from vecl.utils.downloader import download_from_s3, extract_tar_file


def prepare_dataset_configs(
    config: AppConfig,
) -> list[BaseDatasetConfig]:
    """Reads the main manifest, splits it by language, and creates dataset configs."""

    _download_dataset_from_s3(config)
    return _split_dataset_by_language(config)


def _download_dataset_from_s3(config: AppConfig) -> None:
    dataset_base_path = config.paths.dataset_path
    metadata_file = config.paths.metadata_file
    main_metadata_path = dataset_base_path / metadata_file
    s3_bucket_name = config.s3.bucket_name if config.s3 else None
    s3_data_key = config.s3.data_key if config.s3 else None

    if not main_metadata_path.exists():
        if s3_bucket_name:
            print(
                f'Metadata not found at {main_metadata_path}. Attemping to download from S3...'
            )
            try:
                local_tar_path = dataset_base_path / 'processed_data.tar.gz'
                download_from_s3(s3_bucket_name, s3_data_key, local_tar_path)
                extract_tar_file(local_tar_path, dataset_base_path)

                extracted_dir_name = Path(s3_data_key).name.replace(
                    '.tar.gz', ''
                )
                parent_dir = dataset_base_path / extracted_dir_name
                if parent_dir.is_dir():
                    for item_path in parent_dir.iterdir():
                        shutil.move(
                            str(item_path),
                            str(dataset_base_path / item_path.name),
                        )
                    parent_dir.rmdir()
                assert main_metadata_path.exists(), (
                    'Metadata still not found after download.'
                )
            except Exception as e:
                print(f'Error downloading from S3: {e}')
                raise
        else:
            raise FileNotFoundError(
                f'Metadata file not found at {main_metadata_path} and no S3 bucket provided'
            )


def _split_dataset_by_language(config: AppConfig) -> list[BaseDatasetConfig]:
    main_metadata_path = config.paths.dataset_path / config.paths.metadata_file
    main_df = pd.read_csv(main_metadata_path, sep='|').apply(
        lambda x: x.str.strip() if x.dtype == 'object' else x
    )

    dataset_configs = []
    for lang in main_df['language'].dropna().unique():
        lang_df = main_df[main_df['language'] == lang].copy()
        temp_meta_path = config.paths.dataset_path / f'metadata_{lang}.csv'
        lang_df['speaker_name_combined'] = lang_df.apply(
            lambda r: f'{r["dataset"]}_{r["speaker_code"]}_{r.get("speaker_gender", "")}'.rstrip(
                '_'
            ),
            axis=1,
        )

        output_df = pd.DataFrame({
            'full_path': 'audio/' + lang_df['filename'],
            'ignored': 'ignore',
            'text': lang_df['normalized_transcription'],
            'speaker_id': lang_df['speaker_name_combined'],
        })
        output_df.to_csv(temp_meta_path, sep='|', header=False, index=False)

        dataset_configs.append(
            BaseDatasetConfig(
                formatter='brspeech',
                dataset_name=f'multilingual_custom_{lang}',
                meta_file_train=str(temp_meta_path.name),
                path=str(config.paths.dataset_path),
                language=lang,
            )
        )
    return dataset_configs
