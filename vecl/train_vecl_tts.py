from pathlib import Path
from typing import List

import torch
from trainer import TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

# --- VECL-TTS specific imports ------------------------------------------------
from vecl.vecl.config import VeclConfig
from vecl.vecl.vecl import Vecl
from vecl.yourtts.dataset import (
    compute_speaker_embeddings,
    prepare_dataset_configs,
)
from vecl.yourtts.s3_trainer import S3Trainer

# ==================================================================================
# 1. USER ADJUSTABLE PATHS AND TRAINING CONSTANTS
# ==================================================================================

# --- Local filesystem paths -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # path to `vecl/`
WORKSPACE_ROOT = PROJECT_ROOT.parent  # repository root

DATASET_BASE_PATH = (
    WORKSPACE_ROOT / 'data' / 'processed_24k'
)  # contains /audio & metadata.csv
MAIN_METADATA_PATH = DATASET_BASE_PATH / 'metadata.csv'

SPEAKER_EMBEDDINGS_PATH = (
    WORKSPACE_ROOT / 'data' / 'speaker_embeddings_patch.pth'
)
EMOTION_EMBEDDINGS_PATH = WORKSPACE_ROOT / 'data' / 'emotion_embeddings.pth'

# Where VECL-TTS fine-tune outputs (checkpoints, logs) will be stored
OUTPUT_PATH = WORKSPACE_ROOT / 'outputs' / 'vecl_finetune'

# Path to the multilingual CML-TTS checkpoint you want to start from
PRETRAINED_CHECKPOINT_DIR = (
    WORKSPACE_ROOT / 'models' / 'checkpoints_yourtts_cml_tts_dataset'
)
RESTORE_PATH = PRETRAINED_CHECKPOINT_DIR / 'best_model.pth'
LANGUAGE_IDS_JSON = PRETRAINED_CHECKPOINT_DIR / 'language_ids.json'

# --- S3 settings ------------------------------------------------------------------
S3_BUCKET_NAME = 'hotmart-datascience-sagemaker'
S3_PREFIX = 'tts/vecl-tts/checkpoints'

# --- Training hyper-parameters ----------------------------------------------------
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 1e-4
SAVE_STEP = 1000
NUM_LOADER_WORKERS = 4

# ==================================================================================
# 2. HELPER : PATCH LEGACY STATE-DICT WEIGHT_NORM KEYS
# ==================================================================================


def patch_state_dict(state_dict: dict) -> dict:
    """Map legacy `weight_g/weight_v` keys (pre-0.9 Coqui TTS) to PyTorch
    parametrization-compatible keys used in modern versions.
    """
    patched = {}
    for k, v in state_dict.items():
        if '.weight_g' in k:
            new_k = k.replace(
                '.weight_g', '.parametrizations.weight.original0'
            )
        elif '.weight_v' in k:
            new_k = k.replace(
                '.weight_v', '.parametrizations.weight.original1'
            )
        else:
            new_k = k
        patched[new_k] = v
    return patched


# ==================================================================================
# 3. MAIN ENTRY POINT
# ==================================================================================


if __name__ == '__main__':
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 3.1  Build dataset configs for the languages we care about
    # ---------------------------------------------------------------
    dataset_configs: List[BaseDatasetConfig] = prepare_dataset_configs(
        MAIN_METADATA_PATH, DATASET_BASE_PATH
    )

    # Optional: restrict to subset of languages (pt-br & en) --------------
    LANGUAGES_TO_KEEP = {'pt-br', 'en'}
    dataset_configs = [
        cfg for cfg in dataset_configs if cfg.language in LANGUAGES_TO_KEEP
    ]

    # ---------------------------------------------------------------
    # 3.2  Compute (or download) speaker embeddings if missing
    # ---------------------------------------------------------------
    compute_speaker_embeddings(
        dataset_configs,
        SPEAKER_EMBEDDINGS_PATH,
        WORKSPACE_ROOT / 'speaker_encoder_model',
    )

    # ---------------------------------------------------------------
    # 3.3  Assemble VeclConfig
    # ---------------------------------------------------------------
    config = VeclConfig()
    # --- attach datasets & basic training params
    config.datasets = dataset_configs
    config.output_path = str(OUTPUT_PATH)
    config.batch_size = BATCH_SIZE
    config.eval_batch_size = EVAL_BATCH_SIZE
    config.epochs = EPOCHS
    config.learning_rate = LEARNING_RATE
    config.save_step = SAVE_STEP
    config.num_loader_workers = NUM_LOADER_WORKERS

    # --- audio : inherit from CML config (24 kHz)
    config.audio.sample_rate = 24_000
    config.audio.hop_length = 256  # matches CML & YourTTS

    # --- language embedding (num_languages == len(unique))
    config.model_args.use_language_embedding = True
    config.model_args.num_languages = len({
        cfg.language for cfg in dataset_configs
    })
    config.model_args.language_ids_file = str(LANGUAGE_IDS_JSON)

    # --- multi-speaker settings
    config.model_args.use_d_vector_file = True
    config.model_args.d_vector_file = [str(SPEAKER_EMBEDDINGS_PATH)]
    config.model_args.d_vector_dim = 512

    # --- emotion fusion
    config.model_args.emotion_embedding_file = str(EMOTION_EMBEDDINGS_PATH)
    config.model_args.emotion_embedding_dim = 1024

    # ---------------------------------------------------------------
    # 3.4  Prepare data lists for trainer convenience
    # ---------------------------------------------------------------
    train_samples, eval_samples = load_tts_samples(
        datasets=config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # ---------------------------------------------------------------
    # 3.5  Instantiate the VECL model & patch checkpoint
    # ---------------------------------------------------------------
    print('\n>>> Initializing VECL model from config…')
    model = Vecl.init_from_config(config, samples=train_samples)

    if RESTORE_PATH.exists():
        print(f'\n>>> Loading weights from CML-TTS checkpoint: {RESTORE_PATH}')
        checkpoint = torch.load(RESTORE_PATH, map_location='cpu')
        patched_state_dict = patch_state_dict(checkpoint['model'])
        missing, unexpected = model.load_state_dict(
            patched_state_dict, strict=False
        )
        print(
            f'    ✓ Checkpoint loaded with {len(missing)} missing & {len(unexpected)} unexpected keys.'
        )
    else:
        print('⚠️  Pre-trained checkpoint not found. Training from scratch.')

    # ---------------------------------------------------------------
    # 3.6  Kick off training with S3 uploads
    # ---------------------------------------------------------------
    trainer = S3Trainer(
        args=TrainerArgs(),  # defaults are fine; we override via config
        config=config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        s3_bucket=S3_BUCKET_NAME,
        s3_prefix=S3_PREFIX,
    )

    print('\n🚀 Starting VECL-TTS fine-tuning…')
    trainer.fit()
