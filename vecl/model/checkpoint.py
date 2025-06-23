"""
Utilities for loading model checkpoints, patching state dicts, and handling
model-related file downloads.
"""

from pathlib import Path

import torch
from TTS.config import load_config as coqui_load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager

from vecl.config import AppConfig
from vecl.model.layers import EmotionProj
from vecl.model.strategy.factory import get_model_strategy
from vecl.model.vecl import Vecl
from vecl.training.utils import patch_state_dict
from vecl.utils.downloader import download_s3_file


def _resolve_speaker_encoder_paths(config: AppConfig) -> tuple[Path, Path]:
    """
    Ensures the speaker encoder model and config are available, downloading
    them from S3 if necessary.
    """
    se_model_path = config.paths.speaker_encoder_model_dir / 'model_se.pth'
    se_config_path = config.paths.speaker_encoder_model_dir / 'config_se.json'

    if not se_model_path.is_file():
        print('Speaker encoder model not found locally.')
        download_s3_file(
            bucket_name=config.s3.bucket_name,
            s3_key='speaker_encoder/model_se.pth',
            local_path=se_model_path,
        )

    if not se_config_path.is_file():
        print('Speaker encoder config not found locally.')
        download_s3_file(
            bucket_name=config.s3.bucket_name,
            s3_key='speaker_encoder/config_se.json',
            local_path=se_config_path,
        )

    return se_model_path, se_config_path


def load_model_for_training(config: AppConfig, dataset_configs: list):
    """
    Loads a VECL or YourTTS model for training using the application config.
    It handles path resolution, configuration massaging, state dict patching,
    and model initialization.
    """
    model_strategy = get_model_strategy(config)

    # 1. Load the pretrained model config from the checkpoint directory.
    model_config_path = config.paths.restore_path.parent / 'config.json'
    if not model_config_path.exists():
        raise FileNotFoundError(
            f'Pretrained config not found at {model_config_path}'
        )
    model_config = coqui_load_config(model_config_path)

    # Find and set the language_ids.json path, which is critical for
    # initializing the language embedding layer with the correct size.
    language_ids_path = config.paths.restore_path.parent / 'language_ids.json'
    if language_ids_path.is_file():
        print(f'Found language_ids.json at {language_ids_path}')
        model_config.language_ids_file = str(language_ids_path)
        if hasattr(model_config, 'model_args'):
            model_config.model_args.language_ids_file = str(language_ids_path)
    else:
        print(
            f'WARNING: language_ids.json not found alongside the model config at {language_ids_path}. '
            'The model may not initialize correctly if it is multi-lingual.'
        )

    # 2. Patch config using the strategy for the specific model type.
    model_config = model_strategy.patch_config_for_training(
        model_config, dataset_configs
    )

    # 3. Apply all overrides from our main AppConfig.
    model_config.output_path = str(config.paths.output_path)
    model_config.run_name = config.training.run_name
    model_config.logger_uri = None
    model_config.batch_size = config.training.batch_size
    model_config.eval_batch_size = config.training.eval_batch_size
    model_config.num_loader_workers = config.training.num_loader_workers
    model_config.epochs = config.training.epochs
    model_config.lr = config.training.learning_rate
    model_config.save_step = config.training.save_step
    model_config.max_text_len = config.training.max_text_len
    model_config.datasets = dataset_configs

    model_config.audio.sample_rate = config.audio.sample_rate
    model_config.audio.max_audio_len = int(
        config.audio.max_audio_len_seconds * config.audio.sample_rate
    )

    # 4. Initialize the model using the strategy, unpacking the returned tuple
    model, model_config = model_strategy.init_model_from_config(model_config)

    # 5. Load pretrained weights
    restore_path = config.paths.restore_path
    if not restore_path.exists():
        raise FileNotFoundError(f'Restore path not found: {restore_path}')

    checkpoint = torch.load(restore_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    patched_state_dict = patch_state_dict(model_state_dict)

    if (
        not config.training.use_pretrained_lang_embeddings
        and 'emb_l.weight' in patched_state_dict
    ):
        del patched_state_dict['emb_l.weight']

    model.load_state_dict(patched_state_dict, strict=False)

    return model


def load_model_for_inference(
    config: AppConfig,
    device: torch.device,
):
    """
    Loads a VECL or YourTTS model for inference using the application config.
    It handles path resolution, configuration massaging, state dict patching,
    and dynamic layer creation.
    """
    model_strategy = get_model_strategy(config)
    checkpoint_path = model_strategy.get_checkpoint_path_for_inference()

    if not checkpoint_path.is_file():
        # TODO: Add S3 download logic for main model checkpoints
        raise FileNotFoundError(f'Checkpoint not found at: {checkpoint_path}')

    model_config_path = checkpoint_path.parent / 'config.json'
    language_ids_path = checkpoint_path.parent / 'language_ids.json'
    if not language_ids_path.is_file():
        language_ids_path = None

    # 1. Load & massage config
    cfg = coqui_load_config(model_config_path)

    if getattr(cfg, 'audio', None) is None:
        cfg.audio = type(
            'AudioCfg',
            (),
            dict(
                sample_rate=24_000,
                hop_length=256,
                win_length=1024,
                fft_size=1024,
                mel_fmin=0.0,
                mel_fmax=None,
                num_mels=80,
            ),
        )()
    cfg.audio.sample_rate = 24_000
    cfg.model_args.d_vector_file = None
    cfg.model_args.use_speaker_encoder_as_loss = False

    if language_ids_path:
        cfg.language_ids_file = str(language_ids_path)
        cfg.model_args.language_ids_file = str(language_ids_path)
    else:
        cfg.language_ids_file = None
        cfg.model_args.language_ids_file = None

    # Resolve speaker encoder paths, downloading if needed
    se_model_path, se_config_path = _resolve_speaker_encoder_paths(config)
    cfg.model_args.speaker_encoder_model_path = str(se_model_path)
    cfg.model_args.speaker_encoder_config_path = str(se_config_path)

    # 2. Build model shell based on the actual model config file
    is_vecl_model = hasattr(cfg.model_args, 'emotion_embedding_dim')
    print(f'Loading model... (VECL model: {is_vecl_model})')
    if is_vecl_model:
        model = Vecl.init_from_config(cfg)[0]
    else:
        model = setup_model(cfg)

    # 3. Load checkpoint state dict
    sd = torch.load(checkpoint_path, map_location='cpu')['model']
    for key in list(sd.keys()):
        if 'speaker_encoder' in key:
            del sd[key]

    if 'emb_l.weight' in sd:
        pre_shape = sd['emb_l.weight'].shape
        cur_shape = model.emb_l.weight.shape
        if pre_shape != cur_shape:
            print(
                f'⚠️ Language embedding mismatch ({pre_shape} vs {cur_shape}); ignoring.'
            )
            del sd['emb_l.weight']

    if not hasattr(model, 'emotion_proj') or model.emotion_proj is None:
        if 'emotion_proj.proj.weight' in sd:
            w_shape = sd['emotion_proj.proj.weight'].shape
            in_dim, out_dim = w_shape[1], w_shape[0]
            model.emotion_proj = EmotionProj(in_dim, out_dim)
            setattr(model, 'emotion_proj', model.emotion_proj)

    # 4. Patch and load state dict
    sd = patch_state_dict(sd)
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys:
        print('ℹ️ Missing keys:', load_res.missing_keys)
    if load_res.unexpected_keys:
        print('ℹ️ Unexpected keys:', load_res.unexpected_keys)

    # 5. Final setup
    if language_ids_path:
        model.language_manager = LanguageManager(language_ids_path)

    model.eval().to(device)
    return model, cfg
