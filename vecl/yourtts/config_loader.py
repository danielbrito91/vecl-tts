import json
from pathlib import Path
from typing import List, Optional

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import (
    VitsArgs,
    VitsConfig,
)
from TTS.utils.download import download_url

# Optional VECL imports (lazy)
try:
    from vecl.vecl.config import VeclArgs, VeclConfig  # type: ignore
except ImportError:  # during plain YourTTS usage VECL may not be available
    VeclArgs, VeclConfig = None, None

SPEAKER_ENCODER_CONFIG_URL = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json'
SPEAKER_ENCODER_CHECKPOINT_PATH = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar'


def load_config(
    checkpoint_dir: str,
    speaker_encoder_model_dir: Path,
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
    # Custom params
    use_original_lang_ids: bool = False,
    # VECL-TTS params
    emotion_embeddings_file: Optional[str] = None,
    use_emotion_consistency_loss: bool = False,
) -> 'VitsConfig':  # return type is VitsConfig or VeclConfig
    """
    Loads a VitsConfig from a pre-trained model's directory and updates it
    with parameters for fine-tuning on a new dataset.
    """
    print('\n>>> Configuring VITS model for fine-tuning...')

    # --- Speaker Encoder Setup ---
    se_config_path = speaker_encoder_model_dir / 'config_se.json'
    if not se_config_path.exists():
        print(
            f'Downloading speaker encoder config from {SPEAKER_ENCODER_CONFIG_URL} to {speaker_encoder_model_dir}'
        )
        speaker_encoder_model_dir.mkdir(parents=True, exist_ok=True)
        download_url(
            SPEAKER_ENCODER_CONFIG_URL, str(speaker_encoder_model_dir)
        )

    se_model_path = (
        speaker_encoder_model_dir / Path(SPEAKER_ENCODER_CHECKPOINT_PATH).name
    )
    if not se_model_path.exists():
        print(
            f'Downloading speaker encoder model from {SPEAKER_ENCODER_CHECKPOINT_PATH} to {speaker_encoder_model_dir}'
        )
        speaker_encoder_model_dir.mkdir(parents=True, exist_ok=True)
        download_url(
            SPEAKER_ENCODER_CHECKPOINT_PATH, str(speaker_encoder_model_dir)
        )

    # --- Load Base Config Manually ---
    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found at {config_path}.')

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # ------------------------------------------------------------------
    # Decide whether we are in VECL mode (emotion fusion) or plain VITS.
    # ------------------------------------------------------------------
    vecl_mode = bool(emotion_embeddings_file or use_emotion_consistency_loss)

    model_args_cls = (
        VeclArgs if vecl_mode and VeclArgs is not None else VitsArgs
    )
    config_cls = (
        VeclConfig if vecl_mode and VeclConfig is not None else VitsConfig
    )

    model_args_from_json = config_dict.get('model_args', {})
    model_args = model_args_cls()
    for key, value in model_args_from_json.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)

    # Create the main config object with the correctly populated model_args
    config = config_cls(model_args=model_args)

    # Load the rest of the config fields from the original config, ensuring they are objects not dicts
    for key, value in config_dict.items():
        if key not in ['model_args', 'characters'] and hasattr(config, key):
            if (
                hasattr(config, key)
                and isinstance(getattr(config, key), object)
                and isinstance(value, dict)
            ):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(config, key), sub_key):
                        setattr(getattr(config, key), sub_key, sub_value)
            else:
                setattr(config, key, value)

    print(
        ' > ✅ Base config loaded successfully, preserving original settings.'
    )

    # --- THE DEFINITIVE FIX: Use the exact character set from the original recipe ---
    config.characters = CharactersConfig(
        characters_class='TTS.tts.models.vits.VitsCharacters',
        pad='_',
        eos='&',
        bos='*',
        blank=None,
        characters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00a1\u00a3\u00b7\u00b8\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c7\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d9\u00da\u00db\u00dc\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u0101\u0104\u0105\u0106\u0107\u010b\u0119\u0141\u0142\u0143\u0144\u0152\u0153\u015a\u015b\u0161\u0178\u0179\u017a\u017b\u017c\u020e\u04e7\u05c2\u1b20',
        punctuations="\u2014!'(),-.:;?\u00bf ",
        phonemes="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b'\u0303' ",
        is_unique=True,
        is_sorted=True,
    )
    config.model_args.num_chars = len(config.characters)

    # --- Emotion embedding ---------------------------------------------------
    if vecl_mode and emotion_embeddings_file:
        config.model_args.emotion_embedding_file = emotion_embeddings_file
        # default dims set elsewhere; user can override later if needed
    if vecl_mode:
        config.model_args.use_emotion_consistency_loss = (
            use_emotion_consistency_loss
        )

    # --- Language & Speaker Embedding Setup ---
    language_ids_file = None
    num_languages = 0
    if use_original_lang_ids:
        print(' > Using original language IDs from pretrained model.')
        language_ids_file = Path(checkpoint_dir) / 'language_ids.json'
        if not language_ids_file.exists():
            raise FileNotFoundError(
                f'language_ids.json not found in checkpoint_dir: {checkpoint_dir}'
            )
        with open(language_ids_file, 'r', encoding='utf-8') as f:
            language_ids_dict = json.load(f)
        num_languages = len(language_ids_dict)
        print(f' > Number of languages: {num_languages}')
    elif dataset_configs:
        languages = {lang for dc in dataset_configs for lang in dc.languages}
        num_languages = len(languages)
        config.languages = list(languages)

    correct_lang_id_path = (
        (str(language_ids_file)) if language_ids_file else None
    )

    # --- Update Config for Fine-tuning ---
    print(' > Updating config for fine-tuning...')

    # 1. Freeze the text encoder to prevent catastrophic forgetting
    config.model_args.freeze_encoder = True

    # 2. Enable weighted sampling to balance languages in each batch
    config.use_weighted_sampler = True
    config.weighted_sampler_attrs = {'language': 1.0}

    config.model_args.d_vector_file = [str(embeddings_file)]
    config.model_args.use_d_vector_file = True
    config.model_args.speaker_encoder_model_path = str(se_model_path)
    config.model_args.speaker_encoder_config_path = str(se_config_path)
    config.model_args.num_languages = num_languages
    config.model_args.language_ids_file = correct_lang_id_path

    config.output_path = output_path
    config.datasets = dataset_configs
    config.language_ids_file = correct_lang_id_path
    config.epochs = epochs
    config.lr = learning_rate
    config.batch_size = batch_size
    config.eval_batch_size = eval_batch_size
    config.num_loader_workers = num_loader_workers
    config.save_step = save_step
    config.max_text_len = max_text_len
    config.audio.sample_rate = sample_rate
    config.max_audio_len = sample_rate * max_audio_len_seconds

    if use_wandb:
        config.dashboard_logger = 'wandb'
        config.project_name = wandb_project
        config.run_name = wandb_run_name
        config.logger_uri = wandb_entity
    else:
        config.dashboard_logger = 'tensorboard'

    print(' > ✅ VITS config loaded and updated for fine-tuning.')
    return config
