import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from TTS.config.shared_configs import BaseDatasetConfig


@pytest.fixture(autouse=True)
def cleanup_hydra():
    """Fixture to automatically clear Hydra's global state after each test."""
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame for testing."""
    return pd.DataFrame({
        'filename': [
            'audio_001.wav',
            'audio_002.wav',
            'audio_003.wav',
            'audio_004.wav',
        ],
        'language': ['en', 'pt-br', 'en', 'pt-br'],
        'normalized_transcription': [
            'hello world one',
            'olá mundo dois',
            'hello world three',
            'olá mundo quatro',
        ],
        'dataset': ['dataset1', 'dataset1', 'dataset2', 'dataset2'],
        'speaker_code': ['sp001', 'sp002', 'sp001', 'sp003'],
        'speaker_gender': ['M', 'F', 'M', 'F'],
    })


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for dataset testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def create_metadata_file():
    """Helper fixture to create metadata files in test directories."""

    def _create_metadata_file(
        dataset_dir: Path, df: pd.DataFrame, filename: str = 'metadata.csv'
    ):
        """Create a metadata file in the dataset directory."""
        metadata_path = dataset_dir / filename
        df.to_csv(metadata_path, sep='|', index=False)
        return metadata_path

    return _create_metadata_file


# Embedding test fixtures
@pytest.fixture
def mock_dataset_configs():
    """Create mock dataset configs for testing."""
    config1 = MagicMock(spec=BaseDatasetConfig)
    config1.language = 'en'
    config2 = MagicMock(spec=BaseDatasetConfig)
    config2.language = 'pt-br'
    return [config1, config2]


@pytest.fixture
def sample_tts_samples():
    """Create sample TTS samples for testing."""
    return [
        {
            'speaker_name': 'speaker_001',
            'audio_file': '/path/to/audio_001.wav',
            'audio_unique_name': 'audio_001#speaker_001',
            'root_path': '/path/to',
            'text': 'hello world',
        },
        {
            'speaker_name': 'speaker_002',
            'audio_file': '/path/to/audio_002.wav',
            'audio_unique_name': 'audio_002#speaker_002',
            'root_path': '/path/to',
            'text': 'another sentence',
        },
        {
            'speaker_name': 'speaker_001',  # Same speaker, different audio
            'audio_file': '/path/to/audio_003.wav',
            'audio_unique_name': 'audio_003#speaker_001',
            'root_path': '/path/to',
            'text': 'third sentence',
        },
    ]


@pytest.fixture
def mock_audio_tensor():
    """Create a mock audio tensor for testing."""
    return torch.randn(1, 16000)  # 1 second of audio at 16kHz
