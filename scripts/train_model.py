import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer import TrainerArgs
from TTS.tts.datasets import load_tts_samples

from vecl.config import AppConfig
from vecl.data.downloader import create_download_manager
from vecl.data.preparation import DatasetPreparer
from vecl.embeddings.emotion import (
    compute_emotion_embeddings,
    load_emotion_embedder,
)
from vecl.embeddings.speaker import (
    compute_speaker_embeddings,
    embeddings_cover_dataset,
)
from vecl.models.loader import load_model_for_training
from vecl.training.trainer import VeclTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def ensure_artifacts(config: AppConfig):
    """Ensure required artifacts exist."""
    logger.info('Checking required artifacts...')

    # Create download manager from config
    backend_type = None
    backend_config = None
    if config.s3:
        backend_type = 's3'
        backend_config = {'bucket_name': config.s3.bucket_name}

    dm = create_download_manager(config, backend_type, backend_config)

    # Check dataset and model
    dm.get('dataset')
    dm.get('yourtts_checkpoint')

    # Try to get pre-computed embeddings (optional)
    for name in ['speaker_embeddings', 'emotion_embeddings']:
        try:
            dm.get(name)
        except Exception as e:
            logger.info(f'{name} not found, will compute if needed: {e}')


def prepare_data(config: AppConfig):
    """Prepare datasets and load samples."""
    logger.info('Preparing datasets...')

    # Prepare dataset configs
    preparer = DatasetPreparer(
        dataset_path=config.paths.dataset_path,
        sub_dir_name=config.paths.subdir_name,
        metadata_file=config.paths.metadata_file,
    )
    dataset_configs = preparer.prepare_configs()

    # Load samples
    train_samples, eval_samples = load_tts_samples(
        datasets=dataset_configs,
        eval_split=True,
        eval_split_max_size=config.training.eval_split_max_size,
        eval_split_size=config.training.eval_split_size,
    )

    logger.info(
        f'Loaded {len(train_samples)} train, {len(eval_samples)} eval samples'
    )
    return dataset_configs, train_samples, eval_samples


def ensure_embeddings(config: AppConfig, dataset_configs):
    """Compute embeddings if needed."""
    # Speaker embeddings
    if config.training.use_d_vector_file:
        if not config.paths.speaker_embeddings_file.exists():
            logger.info('Computing speaker embeddings...')
            compute_speaker_embeddings(
                dataset_configs=dataset_configs,
                embeddings_file_path=config.paths.speaker_embeddings_file,
                speaker_encoder_model_dir=config.paths.speaker_encoder_model_dir,
            )

        if not embeddings_cover_dataset(
            config.paths.speaker_embeddings_file, dataset_configs
        ):
            logger.error('Speaker embeddings do not cover all audio clips.')
            raise ValueError(
                'Speaker embeddings do not cover all audio clips.'
            )

    # Emotion embeddings (VECL only)
    if config.model.type == 'vecl' and config.training.use_emotion_embedding:
        if not config.paths.emotion_embeddings_file.exists():
            logger.info('Computing emotion embeddings...')

            emotion_embedder = load_emotion_embedder(
                config.model.ser_model_name
            )

            compute_emotion_embeddings(
                dataset_configs=dataset_configs,
                embeddings_file_path=config.paths.emotion_embeddings_file,
                emotion_embedder=emotion_embedder,
            )


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    """Main training entry point."""
    load_dotenv()

    # Convert to AppConfig
    config = AppConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )

    # Setup paths
    config.paths.output_path.mkdir(parents=True, exist_ok=True)
    if config.training.use_d_vector_file and not config.training.d_vector_file:
        config.training.d_vector_file = [config.paths.speaker_embeddings_file]

    logger.info(f'Training {config.model.type} model')
    logger.info(f'Output: {config.paths.output_path}')

    # Step 1: Ensure artifacts exist
    # ensure_artifacts(config)

    # Step 2: Prepare data
    dataset_configs, train_samples, eval_samples = prepare_data(config)

    # Step 3: Ensure embeddings
    ensure_embeddings(config, dataset_configs)

    # Step 4: Load model
    logger.info('Loading model...')
    model = load_model_for_training(config, dataset_configs)

    # Step 5: Train
    logger.info('Starting training...')
    trainer = VeclTrainer(
        args=TrainerArgs(),
        config=model.config,
        output_path=str(config.paths.output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        s3_config=config.s3 if config.s3 else None,
    )

    trainer.fit()
    logger.info('Training complete!')


if __name__ == '__main__':
    main()
