import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer import TrainerArgs
from TTS.tts.datasets import load_tts_samples

from vecl.config import AppConfig
from vecl.dataset.preparation import prepare_dataset_configs
from vecl.embeddings import (
    compute_emotion_embeddings,
    compute_speaker_embeddings,
)
from vecl.model.checkpoint import load_model_for_training
from vecl.model.strategy.factory import get_model_strategy
from vecl.training.trainer import UnifiedTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training script for VECL and YourTTS models.
    """
    load_dotenv()
    # --- 1. Load and Validate Configuration ---
    logger.info('Loading and validating configuration...')
    config = AppConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )
    model_strategy = get_model_strategy(config)

    # CLEAN FIX: Ensure d_vector_file is properly set from paths
    if config.training.use_d_vector_file and not config.training.d_vector_file:
        logger.info(
            '🔧 Setting d_vector_file from paths.speaker_embeddings_file'
        )
        config.training.d_vector_file = [config.paths.speaker_embeddings_file]

    output_path = Path(config.paths.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'Configuration loaded. Output path: {output_path}')
    logger.info(f'Run name: {config.training.run_name}')

    # --- 2. Prepare Dataset Configs and Samples ---
    logger.info('Preparing dataset configurations...')
    dataset_configs = prepare_dataset_configs(config)

    logger.info('Loading dataset samples...')
    train_samples, eval_samples = load_tts_samples(
        datasets=dataset_configs,
        eval_split=True,
        eval_split_max_size=config.training.eval_split_max_size,
        eval_split_size=config.training.eval_split_size,
    )
    logger.info(
        f'Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.'
    )

    # --- 3. Compute Speaker Embeddings (D-Vectors) ---
    if config.training.use_d_vector_file:
        logger.info('Computing speaker embeddings (d-vectors)...')
        compute_speaker_embeddings(
            dataset_configs=dataset_configs,
            embeddings_file_path=config.paths.speaker_embeddings_file,
            speaker_encoder_model_dir=config.paths.speaker_encoder_model_dir,
            s3_bucket=config.s3.bucket_name if config.s3 else None,
            s3_key=config.s3.speaker_embeddings_key if config.s3 else None,
        )
        logger.info('✅ Speaker embeddings computation completed.')
    else:
        logger.info(
            '⚠️ D-vector file usage is disabled. Skipping speaker embeddings computation.'
        )

    # --- 4. Compute Emotion Embeddings ---
    if config.training.use_emotion_embedding and config.model.type == 'vecl':
        logger.info('Computing emotion embeddings...')
        compute_emotion_embeddings(
            dataset_configs=dataset_configs,
            embeddings_file_path=config.paths.emotion_embeddings_file,
            ser_model_name=config.model.ser_model_name,
            s3_bucket=config.s3.bucket_name if config.s3 else None,
            s3_key=config.s3.emotion_embeddings_key if config.s3 else None,
        )
        logger.info('✅ Emotion embeddings computation completed.')

    else:
        logger.info(
            '⚠️ Emotion embeddings disabled or not using VECL model. Skipping emotion embeddings computation.'
        )

    # --- 5. Initialize Model ---
    logger.info(f'Initializing model of type: {config.model.type}')

    model = load_model_for_training(config, dataset_configs)

    # --- 6. Initialize Trainer ---
    logger.info('Initializing trainer...')
    trainer_args = (
        TrainerArgs()
    )  # We can populate this from config later if needed

    # Prepare S3 arguments only if S3 is configured
    s3_kwargs = {}
    if config.s3:
        logger.info(
            'S3 configuration found. Setting up S3 arguments for trainer.'
        )
        s3_kwargs['s3_bucket'] = config.s3.bucket_name
        s3_kwargs['s3_prefix'] = model_strategy.get_checkpoint_prefix_s3()
    else:
        logger.info(
            'No S3 configuration found. Trainer will run in local-only mode.'
        )

    trainer = UnifiedTrainer(
        args=trainer_args,
        config=model.config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        dataset_path=config.paths.dataset_path,
        **s3_kwargs,
    )

    # --- 7. Start Training ---
    logger.info('🚀 Starting training...')
    trainer.fit()
    logger.info('🎉 Training finished.')


if __name__ == '__main__':
    main()
