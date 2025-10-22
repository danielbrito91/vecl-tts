from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for all relevant paths."""

    output_path: Path = Field(
        ..., description='Main directory to save all outputs.'
    )
    dataset_path: Path = Field(
        ..., description='Path to the root of the dataset.'
    )
    subdir_name: str = Field(
        ..., description='Name of the subdirectory in the dataset path.'
    )
    metadata_file: str = Field(
        ..., description='Name of the main metadata file.'
    )
    speaker_embeddings_file: Path = Field(
        ..., description='Path to save computed speaker embeddings.'
    )
    emotion_embeddings_file: Path = Field(
        ..., description='Path to save computed emotion embeddings.'
    )
    speaker_encoder_model_dir: Path = Field(
        ..., description='Directory to store the speaker encoder model.'
    )
    pretrained_checkpoint_dir: Path = Field(
        ...,
        description='Directory to store the downloaded pretrained YourTTS CML model.',
    )
    restore_path: Path = Field(
        ..., description='Path to a checkpoint to restore from.'
    )
    pretrained_config_path: Path = Field(
        ..., description='Path to the pretrained config file.'
    )
    local_tar_path: Path = Field(
        ..., description='Local path to the tar file for S3 upload/download.'
    )
    language_ids_file: Path = Field(
        ...,
        description='Path to the language IDs JSON file for language management.',
    )


class AudioConfig(BaseModel):
    """Configuration for audio processing."""

    sample_rate: int = Field(
        ..., description='Target sample rate for all audio.'
    )
    max_audio_len_seconds: int = Field(
        ..., description='Maximum audio length in seconds to be processed.'
    )


class TrainingConfig(BaseModel):
    """Configuration for training."""

    batch_size: int = Field(..., description='Batch size for training.')
    eval_batch_size: int = Field(..., description='Batch size for evaluation.')
    num_loader_workers: int = Field(
        ..., description='Number of workers for DataLoader.'
    )
    epochs: int = Field(..., description='Number of epochs to train for.')
    learning_rate: float = Field(
        ..., description='Learning rate for the optimizer.'
    )
    save_step: int = Field(..., description='Save a checkpoint every N steps.')
    max_text_len: int = Field(
        ..., description='Maximum number of characters in a text sample.'
    )
    skip_train_epoch: bool = Field(
        ..., description='Skip the training epoch (useful for debugging eval).'
    )
    use_speaker_weighted_sampler: bool = Field(
        ...,
        description='Use a weighted sampler to balance speakers in batches.',
    )
    use_language_weighted_sampler: bool = Field(
        ...,
        description='Use a weighted sampler to balance languages in batches.',
    )
    min_audio_len: int = Field(
        ..., description='Minimum audio length in samples.'
    )
    max_audio_len: int = Field(
        ..., description='Maximum audio length in samples.'
    )
    text_cleaners: List[str] = Field(
        ..., description='List of text cleaners to apply.'
    )
    use_phonemes: bool = Field(
        ..., description='Use phonemes instead of characters.'
    )
    use_precomputed_embeddings: bool = Field(
        ..., description='Use precomputed speaker and emotion embeddings.'
    )
    use_speaker_embedding: bool = Field(
        ..., description='Use speaker embeddings during training.'
    )
    use_emotion_embedding: bool = Field(
        ..., description='Use emotion embeddings during training.'
    )
    use_d_vector_file: bool = Field(
        ..., description='Use d-vector files for speaker embeddings.'
    )
    d_vector_file: Optional[List[Path]] = Field(
        None, description='List of paths to d-vector files (if used).'
    )
    use_multi_lingual: bool = Field(
        ..., description='Enable multi-lingual training.'
    )
    use_pretrained_lang_embeddings: bool = Field(
        ..., description='Use pretrained language embeddings.'
    )
    use_language_embedding: bool = Field(
        True, description='Use language embeddings during training.'
    )
    run_name: str = Field(
        ..., description='Name of the training run for logging.'
    )
    eval_split_max_size: int = Field(
        ..., description='Max number of samples for the evaluation split.'
    )
    eval_split_size: float = Field(
        ..., description='Fraction of the dataset to use for evaluation.'
    )


class S3Config(BaseModel):
    """Configuration for AWS S3 integration."""

    bucket_name: str = Field(..., description='Name of the S3 bucket.')
    data_key: str = Field(..., description='S3 key for the dataset.')
    checkpoint_prefix_yourtts: str = Field(
        ..., description='S3 prefix for YourTTS checkpoints.'
    )
    checkpoint_prefix_vecl: str = Field(
        ..., description='S3 prefix for VECL checkpoints.'
    )
    cml_tts_checkpoint_key: str = Field(
        ..., description='S3 key for the CML-TTS model.'
    )
    emotion_embeddings_key: str = Field(
        'tts/vecl-tts/emotion_embeddings.pth',
        description='S3 key for the pre-computed emotion embeddings file.',
    )
    speaker_embeddings_key: str = Field(
        'tts/yourtts/embeddings/speakers.pth',
        description='S3 key for the pre-computed speaker embeddings file.',
    )


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases."""

    project_name: str = Field(..., description='WandB project name.')
    entity: str = Field(..., description='WandB entity (username or team).')


class ModelConfig(BaseModel):
    """Configuration specific to the model being trained."""

    type: str = Field(
        ..., description="Type of model to train ('vecl' or 'yourtts')."
    )
    ser_model_name: str = Field(
        'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
        description='Name of the Speech Emotion Recognition model.',
    )
    use_emotion_consistency_loss: bool = Field(
        False,
        description='Whether to use the emotion consistency loss (for VECL).',
    )


class DownloadConfig(BaseModel):
    """Configuration for downloading artifacts."""

    artifacts: List[str] = Field(
        ['all'],
        description='Which artifacts to download',
    )
    backend: str = Field('local', description='Storage backend (local or s3)')
    s3_bucket: Optional[str] = Field(
        None,
        description='S3 bucket name (uses S3_BUCKET_NAME env var if not provided)',
    )
    local_mirror: Optional[Path] = Field(
        None, description='Path to local mirror (for local backend)'
    )
    list: bool = Field(False, description='List artifacts and exit')


class AppConfig(BaseModel):
    """Root configuration for the entire application."""

    paths: PathsConfig
    audio: AudioConfig
    training: TrainingConfig
    s3: Optional[S3Config] = None
    wandb: Optional[WandbConfig] = None
    model: ModelConfig
    download: DownloadConfig
