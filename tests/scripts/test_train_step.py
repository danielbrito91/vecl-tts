from unittest.mock import MagicMock, patch

from omegaconf import DictConfig

from scripts.train_model import main as train_main
from vecl.config import AppConfig


@patch('scripts.train_model.load_dotenv')
@patch('scripts.train_model.get_model_strategy')
@patch('scripts.train_model.prepare_dataset_configs')
@patch('scripts.train_model.load_tts_samples')
@patch('scripts.train_model.compute_speaker_embeddings')
@patch('scripts.train_model.compute_emotion_embeddings')
@patch('scripts.train_model.load_model_for_training')
@patch('scripts.train_model.UnifiedTrainer')
def test_main_training_pipeline_full_run(
    mock_unified_trainer,
    mock_load_model,
    mock_compute_emotion,
    mock_compute_speaker,
    mock_load_samples,
    mock_prepare_configs,
    mock_get_strategy,
    mock_load_dotenv,
    hydra_test_config: DictConfig,
    sample_tts_samples,
    mock_dataset_configs,
):
    """
    Test the main training script, ensuring all components are called correctly
    for a full VECL model training run with speaker and emotion embeddings.
    """
    # --- Setup Mocks ---
    mock_prepare_configs.return_value = mock_dataset_configs
    mock_load_samples.return_value = (sample_tts_samples, sample_tts_samples)

    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_load_model.return_value = mock_model

    mock_trainer_instance = MagicMock()
    mock_unified_trainer.return_value = mock_trainer_instance

    mock_strategy = MagicMock()
    mock_strategy.get_checkpoint_prefix_s3.return_value = 's3_prefix'
    mock_get_strategy.return_value = mock_strategy

    cfg = hydra_test_config

    # --- Execute ---
    train_main(cfg)

    # --- Verifications ---
    mock_load_dotenv.assert_called_once()

    # The script validates the DictConfig into an AppConfig instance.
    # Assert that our mocks were called with this validated AppConfig instance.
    mock_get_strategy.assert_called_once()
    validated_config = mock_get_strategy.call_args[0][0]
    assert isinstance(validated_config, AppConfig)

    # Check that the output directory was created by the script
    assert validated_config.paths.output_path.is_dir()

    mock_prepare_configs.assert_called_once_with(validated_config)
    mock_load_samples.assert_called_once_with(
        datasets=mock_dataset_configs,
        eval_split=True,
        eval_split_max_size=validated_config.training.eval_split_max_size,
        eval_split_size=validated_config.training.eval_split_size,
    )

    # Check that d_vector_file was updated correctly on the validated config
    assert validated_config.training.d_vector_file == [
        validated_config.paths.speaker_embeddings_file
    ]

    mock_compute_speaker.assert_called_once_with(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=validated_config.paths.speaker_embeddings_file,
        speaker_encoder_model_dir=validated_config.paths.speaker_encoder_model_dir,
        s3_bucket=validated_config.s3.bucket_name,
        s3_key=validated_config.s3.speaker_embeddings_key,
    )

    mock_compute_emotion.assert_called_once_with(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=validated_config.paths.emotion_embeddings_file,
        ser_model_name=validated_config.model.ser_model_name,
        s3_bucket=validated_config.s3.bucket_name,
        s3_key=validated_config.s3.emotion_embeddings_key,
    )

    mock_load_model.assert_called_once_with(
        validated_config, mock_dataset_configs
    )

    mock_unified_trainer.assert_called_once()
    _, trainer_kwargs = mock_unified_trainer.call_args
    assert trainer_kwargs['output_path'] == str(
        validated_config.paths.output_path
    )
    assert trainer_kwargs['model'] == mock_model
    assert trainer_kwargs['train_samples'] == sample_tts_samples
    assert trainer_kwargs['eval_samples'] == sample_tts_samples
    assert (
        trainer_kwargs['dataset_path'] == validated_config.paths.dataset_path
    )
    assert trainer_kwargs['s3_bucket'] == validated_config.s3.bucket_name
    assert trainer_kwargs['s3_prefix'] == 's3_prefix'

    mock_trainer_instance.fit.assert_called_once()
