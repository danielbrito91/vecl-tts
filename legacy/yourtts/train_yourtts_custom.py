import os
from pathlib import Path

import pandas as pd
import torch
from trainer import TrainerArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from vecl.yourtts.config_loader import load_config
from vecl.yourtts.dataset import (
    compute_speaker_embeddings,
    prepare_dataset_configs,
)
from vecl.yourtts.s3_trainer import S3Trainer

from vecl.utils.downloader import download_from_s3, extract_tar_file

# =================================================================================================
# 1. Configuration Settings
# =================================================================================================

# --- Paths and Directories ---
# Where to save model checkpoints, logs, and other outputs
OUTPUT_PATH = '/mnt/sagemaker-nvme/tts-checkpoints-multilingual'
DATASET_PATH = Path('/mnt/sagemaker-nvme/tts-dataset')  # Path('data')
METADATA_FILE = 'metadata.csv'
EMBEDDINGS_FILE = DATASET_PATH / 'speakers.pth'

# --- Training Hyperparameters ---
BATCH_SIZE = 12
EVAL_BATCH_SIZE = 6
NUM_LOADER_WORKERS = 8
EPOCHS = 500
LEARNING_RATE = 1e-5
SAVE_STEP = 5000
MAX_TEXT_LEN = 200

# --- Audio Parameters ---
SAMPLE_RATE = 24000
MAX_AUDIO_LEN_SECONDS = 15

SPEAKER_ENCODER_MODEL_DIR = Path(OUTPUT_PATH) / 'speaker_encoder'

# --- S3 Configuration for Checkpoint Uploads ---
S3_BUCKET_NAME = 'hotmart-datascience-sagemaker'
S3_CHECKPOINT_PREFIX = 'tts/yourtts/yourtts-multilingual-checkpoints-finetuned'
S3_CML_TTS_CHECKPOINT_KEY = (
    'tts/cml-tts/checkpoints_yourtts_cml_tts_dataset.tar.bz'
)

# --- Fine-tuning Configuration ---
LOCAL_TAR_PATH = f'/mnt/sagemaker-nvme/{Path(S3_CML_TTS_CHECKPOINT_KEY).name}'
PRETRAINED_CHECKPOINT_DIR = (
    '/mnt/sagemaker-nvme/tts-checkpoints-multilingual/pretrained_yourtts_cml'
)
RESTORE_PATH = str(
    Path(PRETRAINED_CHECKPOINT_DIR)
    / 'checkpoints_yourtts_cml_tts_dataset'
    / 'best_model.pth'
)
SKIP_TRAIN_EPOCH = False

# --- W&B Logging ---
USE_WANDB = True
WANDB_PROJECT = 'yourtts-finetuned'
WANDB_ENTITY = 'danielbrito'
WANDB_RUN_NAME = (
    f'yourtts-finetuned-run-{pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")}'
)
USE_PRETRAINED_LANG_EMBEDDINGS = True


def patch_state_dict(state_dict):
    """
    Patches a state_dict from an older Coqui TTS version to match the new format
    for weight-normalized layers.
    Maps '...weight_g' -> '...parametrizations.weight.original0'
    Maps '...weight_v' -> '...parametrizations.weight.original1'
    """
    print('    > Patching state dictionary for version compatibility...')
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'weight_g' in k:
            new_k = k.replace(
                '.weight_g', '.parametrizations.weight.original0'
            )
        elif 'weight_v' in k:
            new_k = k.replace(
                '.weight_v', '.parametrizations.weight.original1'
            )
        else:
            new_k = k
        new_state_dict[new_k] = v
    print('    > Patching complete.')
    return new_state_dict


if __name__ == '__main__':
    # --- Setup ---
    Path(OUTPUT_PATH).mkdir(exist_ok=True)

    checkpoint_config_path = (
        Path(PRETRAINED_CHECKPOINT_DIR)
        / 'checkpoints_yourtts_cml_tts_dataset'
        / 'config.json'
    )
    if not checkpoint_config_path.exists():
        print(
            f'Pre-trained checkpoint config file not found at {checkpoint_config_path}'
        )
        print('Attempting to download from S3...')
        try:
            download_from_s3(
                S3_BUCKET_NAME, S3_CML_TTS_CHECKPOINT_KEY, LOCAL_TAR_PATH
            )
            print('  > ✅ Download complete.')
            print(
                f'Extracting {LOCAL_TAR_PATH} to {PRETRAINED_CHECKPOINT_DIR}...'
            )

            extract_tar_file(LOCAL_TAR_PATH, PRETRAINED_CHECKPOINT_DIR)
            print('  > ✅ Extraction complete.')

            os.remove(LOCAL_TAR_PATH)
            print('  > ✅ Temporary file removed.')

        except Exception as e:
            print(f'  > ❌ Error downloading from S3: {e}')
            raise
    else:
        print(
            f'  > ✅ Pre-trained checkpoint config file found at {checkpoint_config_path}'
        )

    if USE_WANDB:
        try:
            import wandb

            wandb.login()
            print('✅ Login to Weights & Biases successful.')
        except Exception as e:
            print(
                f'⚠️ Could not log in to W&B: {e}. Defaulting to TensorBoard.'
            )
            USE_WANDB = False

    # --- Data and Embeddings ---
    dataset_configs = prepare_dataset_configs(
        DATASET_PATH / METADATA_FILE, DATASET_PATH
    )
    compute_speaker_embeddings(
        dataset_configs, EMBEDDINGS_FILE, SPEAKER_ENCODER_MODEL_DIR
    )

    # --- Model Configuration ---
    config = load_config(
        checkpoint_dir=PRETRAINED_CHECKPOINT_DIR
        + '/checkpoints_yourtts_cml_tts_dataset',
        speaker_encoder_model_dir=SPEAKER_ENCODER_MODEL_DIR,
        output_path=OUTPUT_PATH,
        dataset_configs=dataset_configs,
        embeddings_file=EMBEDDINGS_FILE,
        # Training params
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=NUM_LOADER_WORKERS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_step=SAVE_STEP,
        max_text_len=MAX_TEXT_LEN,
        # Audio params
        sample_rate=SAMPLE_RATE,
        max_audio_len_seconds=MAX_AUDIO_LEN_SECONDS,
        # Logging params
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_entity=WANDB_ENTITY,
        use_original_lang_ids=USE_PRETRAINED_LANG_EMBEDDINGS,
    )

    # --- Initialization and Training ---
    print('\n>>> Initializing model and trainer...')
    train_samples, eval_samples = load_tts_samples(
        datasets=config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Use the custom Vits class that has the fix
    model = Vits.init_from_config(config)

    print('\n>>> Manually loading and patching checkpoint...')
    checkpoint = torch.load(RESTORE_PATH, map_location='cpu')

    # Get the model's state dict and patch it
    model_state_dict = checkpoint['model']

    if (
        USE_PRETRAINED_LANG_EMBEDDINGS
        and 'emb_l.weight' in model_state_dict
        and model_state_dict['emb_l.weight'].shape[0]
        != config.model_args.num_languages
    ):
        raise ValueError(
            f'The number of languages in the model ({config.model_args.num_languages}) does not match the checkpoint ({model_state_dict["emb_l.weight"].shape[0]}). '
            'Please check your configuration.'
        )

    # If not using pretrained embeddings, we must remove the old ones
    if (
        not USE_PRETRAINED_LANG_EMBEDDINGS
        and 'emb_l.weight' in model_state_dict
    ):
        print(
            '    > Removing language embedding layer from checkpoint because we are not using pretrained embeddings.'
        )
        del model_state_dict['emb_l.weight']

    patched_state_dict = patch_state_dict(model_state_dict)

    # Load the patched state dict into the model
    model.load_state_dict(patched_state_dict, strict=False)
    print('✅ Patched state dictionary loaded into the model.')

    # We pass `restore_path=None` because we have already manually loaded the model.
    trainer = S3Trainer(
        args=TrainerArgs(
            restore_path=None,
            skip_train_epoch=SKIP_TRAIN_EPOCH,
        ),
        config=config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        s3_bucket=S3_BUCKET_NAME,
        s3_prefix=S3_CHECKPOINT_PREFIX,
    )

    print('\n🚀 Starting training...')
    trainer.fit()
