import os

import numpy as np
import soundfile as sf
import torch
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis

USE_CUDA = False
SE_CHECKPOINT_FILEPATH = (
    'models/checkpoints_yourtts_cml_tts_dataset/model_se.pth'
)
SE_CONFIG_FILEPATH = (
    'models/checkpoints_yourtts_cml_tts_dataset/config_se.json'
)
i = '900'
j = 900
FINETUNE_CHECKPOINT = f'models/finetune/checkpoint_{j}.pth'
FINETUNE_CONFIG = f'models/finetune/config_{i}.json'

CML_CHECKPOINT = 'models/checkpoints_yourtts_cml_tts_dataset/best_model.pth'
CONFIG = 'models/checkpoints_yourtts_cml_tts_dataset/config.json'
LANGUAGE_EMBEDDINGS = (
    'models/checkpoints_yourtts_cml_tts_dataset/language_ids.json'
)


def load_model_config(
    checkpoint_filepath,
    config_filepath,
    language_embeddings_filepath,
    use_cuda=True,
    is_finetune=False,
) -> tuple:
    """
    Load a model from a checkpoint and config file, handling the differences
    between a pre-trained and a fine-tuned model.
    """
    print(f'>>> Loading model from checkpoint. is_finetune: {is_finetune}')

    # Load the config
    config = load_config(config_filepath)

    # CRITICAL FIX: Ensure audio config matches training
    if not hasattr(config, 'audio') or config.audio is None:
        config.audio = type(
            'AudioConfig',
            (),
            {
                'sample_rate': 24000,
                'hop_length': 256,
                'win_length': 1024,
                'fft_size': 1024,
                'mel_fmin': 0.0,
                'mel_fmax': None,
                'num_mels': 80,
            },
        )()

    # Ensure sample rate is correct
    config.audio.sample_rate = 24000

    # If it's a fine-tuned model, overwrite the absolute paths
    if is_finetune:
        config.model_args['speaker_encoder_config_path'] = SE_CONFIG_FILEPATH
        config.model_args['speaker_encoder_model_path'] = (
            SE_CHECKPOINT_FILEPATH
        )
        config.model_args['language_ids_file'] = LANGUAGE_EMBEDDINGS
        config.language_ids_file = LANGUAGE_EMBEDDINGS

    # These are set to None for inference
    config.model_args['d_vector_file'] = None
    config.model_args['language_ids_file'] = None
    config.model_args['use_speaker_encoder_as_loss'] = False

    # Setup the model from the config
    model = setup_model(config)

    # Load the checkpoint
    cp = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
    model_weights = cp['model'].copy()

    # Remove speaker encoder weights
    for key in list(model_weights.keys()):
        if 'speaker_encoder' in key:
            del model_weights[key]

    # Handle language embedding mismatch for fine-tuned models
    if is_finetune and 'emb_l.weight' in model_weights:
        # Check if dimensions match
        pretrained_lang_emb_shape = model_weights['emb_l.weight'].shape
        current_lang_emb_shape = model.emb_l.weight.shape

        if pretrained_lang_emb_shape != current_lang_emb_shape:
            print(
                f' > Language embedding mismatch: {pretrained_lang_emb_shape} vs {current_lang_emb_shape}'
            )
            print(
                ' > Not loading language embeddings from fine-tuned checkpoint.'
            )
            del model_weights['emb_l.weight']

    # Patch the state dict for compatibility
    patched_model_weights = patch_state_dict(model_weights)

    # Load the patched weights
    load_result = model.load_state_dict(patched_model_weights, strict=False)
    if load_result.missing_keys:
        print(' > ⚠️ Missing keys:', load_result.missing_keys)
    if load_result.unexpected_keys:
        print(' > ⚠️ Unexpected keys:', load_result.unexpected_keys)

    # Set up the language manager
    model.language_manager = LanguageManager(language_embeddings_filepath)
    model.eval()

    if use_cuda:
        model = model.cuda()

    return model, config


def synthesize_waveform(
    model,
    config,
    sentence,
    reference_emb,
    speed=1.0,
    use_cuda=True,
    language_name='pt-br',
    use_griffin_lim=False,
) -> np.array:
    """
    Run inference on the model with proper settings.
    """

    # Use settings that match the original model's training
    model.length_scale = 2 - speed  # Match original (was 2.0 - speed)
    model.inference_noise_scale = 0.3  # Match original (was 0.333)
    model.inference_noise_scale_dp = 0.3  # Match original (was 0.333)

    # Verify language exists
    if language_name not in model.language_manager.name_to_id:
        available_langs = list(model.language_manager.name_to_id.keys())
        raise ValueError(
            f"Language '{language_name}' not found. Available: {available_langs}"
        )

    language_id = model.language_manager.name_to_id[language_name]
    print(language_id)

    waveform, alignment, _, _ = synthesis(
        model=model,
        text=sentence,
        CONFIG=config,
        use_cuda=use_cuda,
        speaker_id=None,
        style_wav=None,
        style_text=None,
        use_griffin_lim=use_griffin_lim,
        d_vector=reference_emb,
        language_id=language_id,
    ).values()

    return waveform


# Updated save function with correct sample rate
def save_file(
    filename, audio, output_type='wav', sampling_rate=24000
) -> None:  # Changed to 24000
    """
    Save wav or ogg output file with correct sample rate
    """
    if output_type == 'wav':
        sf.write(filename + '.wav', audio, sampling_rate, 'PCM_16')
    else:
        sf.write(
            filename + '.ogg',
            audio,
            sampling_rate,
            format='ogg',
            subtype='vorbis',
        )


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


def extract_reference_embedding(se_speaker_manager, wav_filepath, use_cuda):
    reference_emb = se_speaker_manager.compute_embedding_from_clip(
        wav_filepath
    )
    return reference_emb


def generate(
    sentence: str,
    ref_wav_filepath: str,
    target_language: str,
    output_file_path: str,
    speed: float = 1.0,
    is_finetune: bool = False,
) -> None:
    if is_finetune:
        checkpoint_path = FINETUNE_CHECKPOINT
        config_path = FINETUNE_CONFIG
        language_embeddings_path = os.path.join(
            os.path.dirname(CML_CHECKPOINT), 'language_ids.json'
        )
    else:
        checkpoint_path = CML_CHECKPOINT
        config_path = CONFIG
        language_embeddings_path = LANGUAGE_EMBEDDINGS

    model, config = load_model_config(
        checkpoint_path,
        config_path,
        language_embeddings_path,
        USE_CUDA,
        is_finetune=is_finetune,
    )
    se_speaker_manager = SpeakerManager(
        encoder_model_path=SE_CHECKPOINT_FILEPATH,
        encoder_config_path=SE_CONFIG_FILEPATH,
        use_cuda=USE_CUDA,
    )

    reference_emb = extract_reference_embedding(
        se_speaker_manager, ref_wav_filepath, USE_CUDA
    )
    waveform = synthesize_waveform(
        model,
        config,
        sentence,
        reference_emb,
        speed,
        USE_CUDA,
        target_language,
    )

    save_file(output_file_path, waveform, 'wav', sampling_rate=24000)
