from pathlib import Path
from typing import List

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig


def load_and_update_config(
    checkpoint_dir: str,
    output_path: str,
    dataset_configs: List[BaseDatasetConfig],
    embeddings_file: str,
    # Training params
    batch_size: int,
    eval_batch_size: int,
    num_loader_workers: int,
    epochs: int,
    learning_rate: float,
    save_step: int,
    max_text_len: int,
    # Audio params
    sample_rate: int,
    max_audio_len_seconds: int,
    # Logging params
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str,
    wandb_entity: str,
) -> VitsConfig:
    """
    Loads a VitsConfig from a pre-trained model's directory and updates it
    with parameters for fine-tuning on a new dataset.
    """
    print('\n>>> Configuring VITS model for fine-tuning...')

    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found at {config_path}.')

    # Load the base config from the checkpoint
    config = VitsConfig()
    config.load_json(str(config_path))

    # --- Force Override of Critical Architectural Parameters ---
    # This ensures the model structure matches the YourTTS checkpoint exactly.
    print('    > Forcing YourTTS architectural parameters for compatibility.')
    config.model_args.use_d_vector_file = True
    config.model_args.use_language_embedding = True
    config.model_args.d_vector_dim = 512
    config.model_args.spec_segment_size = 62
    config.model_args.resblock_type_decoder = (
        '2'  # This is a common source of error
    )

    print('    > Forcing Text Encoder parameters.')
    config.model_args.num_layers_text_encoder = 10
    config.model_args.hidden_channels_ffn_text_encoder = 768
    config.model_args.num_heads_text_encoder = 2
    config.model_args.dropout_p_text_encoder = 0.1

    # --- Update Paths and Data Settings ---
    config.output_path = output_path
    config.datasets = dataset_configs
    config.model_args.d_vector_file = [str(embeddings_file)]

    # --- Update Logging, Training, and Audio Settings ---
    config.dashboard_logger = 'wandb' if use_wandb else 'tensorboard'
    config.project_name = wandb_project if use_wandb else None
    config.run_name = wandb_run_name if use_wandb else None
    config.logger_uri = wandb_entity if use_wandb else None
    config.batch_size = batch_size
    config.eval_batch_size = eval_batch_size
    config.num_loader_workers = num_loader_workers
    config.epochs = epochs
    config.lr = learning_rate
    config.save_step = save_step
    config.max_text_len = max_text_len
    config.mixed_precision = True
    config.cudnn_benchmark = True
    config.max_audio_len = sample_rate * max_audio_len_seconds
    config.audio.sample_rate = sample_rate

    print('✅ VITS configuration updated for fine-tuning.')
    return config


# Full train config
# print('\n>>> Configuring VITS model...')
# main_df = pd.read_csv(DATASET_PATH / METADATA_FILE, sep='|')
# punctuations = '!(),-.:;? '
# texts = ''.join(main_df['normalized_transcription'].dropna())
# unique_chars = sorted(list(set(c for c in texts if c not in punctuations)))

# audio_config = VitsAudioConfig(
#     sample_rate=SAMPLE_RATE,
#     hop_length=256,
#     win_length=1024,
#     fft_size=1024,
#     num_mels=80,
# )

# model_args = VitsArgs(
#     d_vector_file=[str(EMBEDDINGS_FILE)],
#     use_d_vector_file=True,
#     d_vector_dim=512,
#     use_language_embedding=True,
# )

# config = VitsConfig(
#     output_path=OUTPUT_PATH,
#     model_args=model_args,
#     audio=audio_config,
#     dashboard_logger='wandb' if USE_WANDB else 'tensorboard',
#     project_name=WANDB_PROJECT if USE_WANDB else None,
#     run_name=WANDB_RUN_NAME if USE_WANDB else None,
#     logger_uri=WANDB_ENTITY if USE_WANDB else None,
#     batch_size=BATCH_SIZE,
#     eval_batch_size=EVAL_BATCH_SIZE,
#     max_text_len=MAX_TEXT_LEN,
#     num_loader_workers=NUM_LOADER_WORKERS,
#     epochs=EPOCHS,
#     lr=LEARNING_RATE,
#     save_step=SAVE_STEP,
#     datasets=dataset_configs,
#     use_phonemes=False,
#     text_cleaner='multilingual_cleaners',
#     characters=CharactersConfig(
#         characters_class='TTS.tts.models.vits.VitsCharacters',
#         pad='_',
#         eos='&',
#         bos='*',
#         blank=None,
#         characters=''.join(unique_chars),
#         punctuations=punctuations,
#     ),
#     max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS,
#     mixed_precision=True,
#     cudnn_benchmark=True,
# )
