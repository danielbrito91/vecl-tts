import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from TTS.config.shared_configs import BaseDatasetConfig

from tests.fixtures.data import SAMPLE_TTS_SAMPLES


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
    """Sample TTS dataset rows used across tests."""
    return SAMPLE_TTS_SAMPLES


@pytest.fixture
def mock_audio_tensor():
    """Create a mock audio tensor for testing."""
    return torch.randn(1, 16000)  # 1 second of audio at 16kHz


@pytest.fixture
def mock_model_outputs():
    """Create a mock model output object with hidden states."""
    mock_outputs = MagicMock()
    # Last hidden state of shape (batch_size, sequence_length, hidden_size)
    mock_outputs.hidden_states = [torch.randn(1, 100, 1024)]
    return mock_outputs


@pytest.fixture
def create_app_config(temp_dataset_dir):
    """Factory to build minimal AppConfig-like stubs for dataset tests."""

    def _factory(
        *,
        dataset_dir=temp_dataset_dir,
        metadata_file='metadata.csv',
        s3_bucket_name=None,
        s3_key='processed_24k.tar.gz',
    ):
        paths = SimpleNamespace(
            dataset_path=dataset_dir, metadata_file=metadata_file
        )
        s3 = (
            SimpleNamespace(bucket_name=s3_bucket_name, data_key=s3_key)
            if s3_bucket_name
            else None
        )
        return SimpleNamespace(paths=paths, s3=s3)

    return _factory


@pytest.fixture
def hydra_test_config(temp_dataset_dir: Path) -> DictConfig:
    """Create a default DictConfig for testing the training script."""
    output_path = temp_dataset_dir / 'output'
    config_dict = {
        'paths': {
            'output_path': str(output_path),
            'dataset_path': str(temp_dataset_dir),
            'metadata_file': 'metadata.csv',
            'speaker_embeddings_file': str(output_path / 'speakers.pth'),
            'emotion_embeddings_file': str(output_path / 'emotions.pth'),
            'speaker_encoder_model_dir': 'speaker_encoder/',
            'pretrained_checkpoint_dir': str(
                output_path / 'pretrained_checkpoints'
            ),
            'restore_path': str(output_path / 'restore.pth'),
            'pretrained_config_path': str(
                output_path / 'pretrained_config.json'
            ),
            'local_tar_path': str(output_path / 'data.tar.gz'),
            'language_ids_file': str(output_path / 'language_ids.json'),
        },
        'audio': {'sample_rate': 24000, 'max_audio_len_seconds': 20},
        'training': {
            'run_name': 'test_run',
            'use_d_vector_file': True,
            'd_vector_file': [],
            'eval_split_max_size': 10,
            'eval_split_size': 0.1,
            'use_emotion_embedding': True,
            'batch_size': 2,
            'eval_batch_size': 2,
            'num_loader_workers': 0,
            'epochs': 1,
            'learning_rate': 1e-5,
            'save_step': 100,
            'max_text_len': 250,
            'skip_train_epoch': False,
            'use_speaker_weighted_sampler': False,
            'use_language_weighted_sampler': False,
            'min_audio_len': 24000,
            'max_audio_len': 24000 * 10,
            'text_cleaners': ['multilingual_cleaners'],
            'use_phonemes': True,
            'use_precomputed_embeddings': True,
            'use_speaker_embedding': True,
            'use_multi_lingual': True,
            'use_pretrained_lang_embeddings': False,
        },
        'model': {
            'type': 'vecl',
            'ser_model_name': 'dummy-ser-model',
            'use_emotion_consistency_loss': False,
        },
        's3': {
            'bucket_name': 'dummy-bucket',
            'emotion_embeddings_key': 's3_emotions.pth',
            'speaker_embeddings_key': 's3_speakers.pth',
            'checkpoint_prefix_yourtts': 'checkpoints/yourtts',
            'checkpoint_prefix_vecl': 'checkpoints/vecl',
            'cml_tts_checkpoint_key': 'checkpoints/cml_tts.pth',
        },
        'wandb': None,
        'datasets': [
            {
                'name': 'test_dataset',
                'path': str(temp_dataset_dir),
                'meta_file_train': 'metadata.csv',
            }
        ],
    }
    return OmegaConf.create(config_dict)
