import logging
from pathlib import Path
from typing import List

import pandas as pd
from TTS.config.shared_configs import BaseDatasetConfig

logger = logging.getLogger(__name__)


class DatasetPreparer:
    def __init__(
        self, dataset_path: Path, metadata_file: str = 'metadata.csv'
    ):
        self.dataset_path = Path(dataset_path)
        self.metadata_file = metadata_file
        self.metadata_path = self.dataset_path / metadata_file

    def prepare_configs(self) -> List[BaseDatasetConfig]:
        """Prepare dataset configurations for each language."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f'Metadata not found at {self.metadata_path}. '
                "Run 'uv run -m scripts.download_artifacts.py --artifacts dataset' first."
            )

        # Read and clean metadata
        metadata = pd.read_csv(self.metadata_path, sep='|')
        metadata = metadata.apply(
            lambda x: x.str.strip() if x.dtype == 'object' else x
        )

        configs = []
        for lang in metadata['language'].unique():
            lang_config = self._create_language_config(metadata, lang)
            configs.append(lang_config)

        logger.info(
            f'Created {len(configs)} dataset configs for languages: {[c.language for c in configs]}'
        )
        return configs

    def _create_language_config(
        self, metadata: pd.DataFrame, language: str
    ) -> BaseDatasetConfig:
        """Create a config for a specific language."""
        lang_df = metadata[metadata['language'] == language].copy()

        # Create speaker names
        lang_df['speaker_name'] = lang_df.apply(
            lambda r: f'{r["dataset"]}_{r["speaker_code"]}_{r.get("speaker_gender", "")}'.rstrip(
                '_'
            ),
            axis=1,
        )

        # Write language-specific metadata
        lang_meta_path = self.dataset_path / f'metadata_{language}.csv'
        output_df = pd.DataFrame({
            'audio_path': 'audio/' + lang_df['filename'],
            'ignored': 'ignore',
            'text': lang_df['normalized_transcription'],
            'speaker_id': lang_df['speaker_name'],
        })
        output_df.to_csv(lang_meta_path, sep='|', header=False, index=False)

        return BaseDatasetConfig(
            formatter='brspeech',
            dataset_name=f'multilingual_custom_{language}',
            meta_file_train=lang_meta_path.name,
            path=str(self.dataset_path),
            language=language,
        )


# def prepare_dataset_configs(config) -> List[BaseDatasetConfig]:
#     preparer = DatasetPreparer(
#         dataset_path=config.paths.dataset_path,
#         metadata_file=config.paths.metadata_file,
#     )
#     return preparer.prepare_configs()
