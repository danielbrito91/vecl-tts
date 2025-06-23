"""
Unified data utilities for VECL-TTS.
This module provides a simple interface to all data processing functionality,
handling configuration extraction and calling the underlying modules.
"""

from TTS.config.shared_configs import BaseDatasetConfig

from vecl.config import AppConfig
from vecl.dataset import VeclDataset

# Import dataset utilities
from vecl.dataset import prepare_dataset_configs as _prepare_dataset_configs
from vecl.embeddings import EmotionEmbedding
from vecl.embeddings import (
    compute_emotion_embeddings as _compute_emotion_embeddings,
)

# Import embedding computation utilities
from vecl.embeddings import (
    compute_speaker_embeddings as _compute_speaker_embeddings,
)
from vecl.evaluation import emotion_consistency
from vecl.model.checkpoint import load_model_for_inference, patch_state_dict
from vecl.model.config import VeclConfig
from vecl.model.layers import EmotionProj
from vecl.model.loss import VeclGeneratorLoss
from vecl.model.vecl import Vecl

# Import preprocessing utilities
from vecl.preprocessing import AudioPreprocessor, TextPreprocessor


# Configuration-aware wrapper functions
def prepare_dataset_configs(config: AppConfig) -> list[BaseDatasetConfig]:
    """Prepare dataset configs using configuration."""
    return _prepare_dataset_configs(
        dataset_base_path=config.paths.dataset_path,
        metadata_file=config.paths.metadata_file,
        s3_bucket_name=config.s3.bucket_name,
    )


def compute_speaker_embeddings(
    config: AppConfig, dataset_configs: list[BaseDatasetConfig]
):
    """Compute speaker embeddings using configuration."""
    return _compute_speaker_embeddings(
        dataset_configs=dataset_configs,
        embeddings_file_path=config.paths.speaker_embeddings_file,
        speaker_encoder_model_dir=config.paths.speaker_encoder_model_dir,
        s3_bucket=config.s3.bucket_name,
        s3_key=config.s3.speaker_embeddings_key,
    )


def compute_emotion_embeddings(
    config: AppConfig, dataset_configs: list[BaseDatasetConfig]
):
    """Compute emotion embeddings using configuration."""
    return _compute_emotion_embeddings(
        dataset_configs=dataset_configs,
        embeddings_file_path=config.paths.emotion_embeddings_file,
        ser_model_name=config.model.ser_model_name,
        s3_bucket=config.s3.bucket_name,
        s3_key=config.s3.emotion_embeddings_key,
    )


# Export all public interfaces
__all__ = [
    # Preprocessing
    'TextPreprocessor',
    'AudioPreprocessor',
    # Embeddings
    'compute_speaker_embeddings',
    'compute_emotion_embeddings',
    'EmotionEmbedding',
    # Evaluation
    'emotion_consistency',
    # Dataset
    'prepare_dataset_configs',
    'VeclDataset',
    # Model Components
    'Vecl',
    'VeclConfig',
    'VeclGeneratorLoss',
    'EmotionProj',
    'load_model_for_inference',
    'patch_state_dict',
]
