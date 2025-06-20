from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    """
    Reads critical environment variables.
    Pydantic will automatically read from a .env file if it exists.
    """

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    OUTPUT_PATH: Path
    DATASET_PATH: Path
    S3_BUCKET_NAME: str
    WANDB_ENTITY: str


# Load the environment variables once
env_config = EnvConfig()


class PathsConfig(BaseModel):
    """Configuration for all relevant paths."""

    output_path: Path = Field(
        default=env_config.OUTPUT_PATH,
        description='Main directory to save all outputs.',
    )
    dataset_path: Path = Field(
        default=env_config.DATASET_PATH,
        description='Path to the root of the dataset.',
    )
    metadata_file: str = Field(
        'metadata.csv', description='Name of the main metadata file.'
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
        description='Directory to store the downloaded pretrained YourTTS model.',
    )
    restore_path: Path = Field(
        ...,
        description='Path to the best_model.pth from the pretrained model.',
    )
    local_tar_path: Path = Field(
        ...,
        description='Temporary local path for downloading S3 archives.',
    )


class AudioConfig(BaseModel):
    """Parameters for audio processing."""

    sample_rate: int = 24000
    max_audio_len_seconds: int = 15


class TrainingConfig(BaseModel):
    """Hyperparameters and settings for training."""

    batch_size: int = 12
    eval_batch_size: int = 6
    num_loader_workers: int = 8
    epochs: int = 500
    learning_rate: float = 1e-5
    save_step: int = 5000
    max_text_len: int = 200
    skip_train_epoch: bool = False
    use_pretrained_lang_embeddings: bool = True


class S3Config(BaseModel):
    """Configuration for S3 bucket interactions."""

    bucket_name: str = Field(
        default=env_config.S3_BUCKET_NAME,
        description='Name of the S3 bucket for cloud storage.',
    )
    checkpoint_prefix_yourtts: str = (
        'tts/yourtts/yourtts-multilingual-checkpoints-finetuned'
    )
    checkpoint_prefix_vecl: str = (
        'tts/vecl/vecl-multilingual-checkpoints-finetuned'
    )
    cml_tts_checkpoint_key: str = (
        'tts/cml-tts/checkpoints_yourtts_cml_tts_dataset.tar.bz'
    )


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging."""

    use_wandb: bool = True
    project_yourtts: str = 'yourtts-finetuned'
    project_vecl: str = 'vecl-finetuned'
    entity: str = Field(
        default=env_config.WANDB_ENTITY,
        description='Your personal or team entity for W&B.',
    )


class ModelConfig(BaseModel):
    """Configuration specific to the model type."""

    type: str = Field(
        'vecl',
        description='Type of model to train. Can be "vecl" or "yourtts".',
    )
    use_emotion_consistency_loss: bool = True


class AppConfig(BaseModel):
    """Root configuration model that ties everything together."""

    paths: PathsConfig
    audio: AudioConfig
    training: TrainingConfig
    s3: S3Config
    wandb: WandbConfig
    model: ModelConfig
