import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
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
FINETUNE_CHECKPOINT = 'models/finetune/checkpoint_73.pth'
FINETUNE_CONFIG = 'models/finetune/config_73.json'

CML_CHECKPOINT = 'models/checkpoints_yourtts_cml_tts_dataset/best_model.pth'
CONFIG = 'models/checkpoints_yourtts_cml_tts_dataset/config.json'
LANGUAGE_EMBEDDINGS = (
    'models/checkpoints_yourtts_cml_tts_dataset/language_ids.json'
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

    # If it's a fine-tuned model, overwrite the absolute paths from the training
    # environment with the correct local paths for inference.
    if is_finetune:
        config.model_args['speaker_encoder_config_path'] = SE_CONFIG_FILEPATH
        config.model_args['speaker_encoder_model_path'] = (
            SE_CHECKPOINT_FILEPATH
        )
        config.model_args['language_ids_file'] = LANGUAGE_EMBEDDINGS
        config.language_ids_file = LANGUAGE_EMBEDDINGS

    # These are set to None for inference, as we use an external speaker encoder
    # and provide the d_vector manually.
    config.model_args['d_vector_file'] = None
    config.model_args['language_ids_file'] = None
    config.model_args['use_speaker_encoder_as_loss'] = False

    # Setup the model from the config
    model = setup_model(config)

    # Load the checkpoint
    cp = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
    model_weights = cp['model'].copy()

    # The speaker encoder is not part of the TTS model checkpoint, so we remove it.
    for key in list(model_weights.keys()):
        if 'speaker_encoder' in key:
            del model_weights[key]

    # For a fine-tuned model, we must handle the language embedding mismatch.
    # In infer.py, inside load_model_config, when is_finetune is True
    if 'emb_l.weight' in model_weights:
        print(
            f'  > Shape of emb_l.weight in checkpoint: {model_weights["emb_l.weight"].shape}'
        )
        print(
            f'  > Shape of model.emb_l.weight after setup_model: {model.emb_l.weight.shape}'
        )
        if model_weights['emb_l.weight'].shape != model.emb_l.weight.shape:
            print(
                ' > Language embedding layer mismatch DETECTED. Deleting from checkpoint weights.'
            )  # Critical log
            del model_weights['emb_l.weight']
        else:
            print(
                ' > Language embedding layer shapes MATCH. Proceeding to load.'
            )
    else:
        print(
            ' > emb_l.weight NOT FOUND in fine-tuned checkpoint model_weights.'
        )

    # After model.load_state_dict:

    # Patch the state dict for compatibility with newer TTS versions.
    patched_model_weights = patch_state_dict(model_weights)

    # Load the patched weights into the model.
    load_result = model.load_state_dict(patched_model_weights, strict=False)
    print(f'  > Load result - Missing keys: {load_result.missing_keys}')
    print(f'  > Load result - Unexpected keys: {load_result.unexpected_keys}')
    if load_result.missing_keys:
        print(
            ' > ⚠️ Missing keys in state_dict loading:',
            load_result.missing_keys,
        )
    if load_result.unexpected_keys:
        print(
            ' > ⚠️ Unexpected keys in state_dict loading:',
            load_result.unexpected_keys,
        )

    # Set up the language manager and put the model in eval mode.
    model.language_manager = LanguageManager(language_embeddings_filepath)
    model.eval()

    if use_cuda:
        model = model.cuda()

    return model, config


def extract_reference_embedding(se_speaker_manager, wav_filepath, use_cuda):
    reference_emb = se_speaker_manager.compute_embedding_from_clip(
        wav_filepath
    )
    return reference_emb


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
    Run inference on the model.
    """

    model.length_scale = (
        2.0 - speed
    )  # scaler for the duration predictor. The larger it is, the slower the speech.
    model.inference_noise_scale = 0.333  # defines the noise variance applied to the random z vector at inference.
    model.inference_noise_scale_dp = 0.333  # defines the noise variance applied to the duration predictor z vector at inference.

    language_id = model.language_manager.name_to_id[language_name]
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


def save_file(filename, audio, output_type='wav', sampling_rate=24000) -> None:
    """
    Save wav or ogg output file
    Args:
        filename: filename without extension (.wav or .ogg)
        audio: numpy data referent to waveform
        output_type: wav or ogg
        sampling_rate: examples 22050, 44100

    Returns: None
    """
    if output_type == 'wav':
        sf.write(filename + '.wav', audio, sampling_rate, 'PCM_16')
    else:
        # A bug occurred in the soundfile lib: cuts the end of the ogg file.
        sf.write(
            filename + '.ogg',
            audio,
            sampling_rate,
            format='ogg',
            subtype='vorbis',
        )


def generate_wavfile(
    model,
    config,
    se_speaker_manager,
    sentences,
    ref_wav_filepath,
    speed=1.0,
    output_folder='output_inference',
    sr=24000,
    audio_format='wav',
    use_cuda=True,
    language_name='pt-br',
) -> None:
    """
    Run inference on the model and save a wav file
    """

    reference_emb = extract_reference_embedding(
        se_speaker_manager, ref_wav_filepath, use_cuda
    )

    for index, sentence in enumerate(tqdm(sentences)):
        waveform = synthesize_waveform(
            model,
            config,
            sentence,
            reference_emb,
            speed,
            use_cuda,
            language_name,
        )
        filename = f'output-{index}'
        filepath = os.path.join(output_folder, filename)
        save_file(filepath, waveform, audio_format, sr)


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


if __name__ == '__main__':
    REFERENCE_WAV = 'data/draft/ptbr/art001a.wav'
    OUTPUT_FOLDER = 'outputs'
    SAMPLE_RATE = 24000
    LANGUAGE = 'pt-br'

    se_speaker_manager = SpeakerManager(
        encoder_model_path=SE_CHECKPOINT_FILEPATH,
        encoder_config_path=SE_CONFIG_FILEPATH,
        use_cuda=USE_CUDA,
    )

    model, config = load_model_config(
        CML_CHECKPOINT, CONFIG, LANGUAGE_EMBEDDINGS, USE_CUDA
    )

    speed = 1.0
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    sentences = [
        'Título um. dos princípios fundamentais da filosofia.',
    ]

    generate_wavfile(
        model,
        config,
        se_speaker_manager,
        sentences,
        REFERENCE_WAV,
        speed,
        OUTPUT_FOLDER,
        SAMPLE_RATE,
        'wav',
        USE_CUDA,
        LANGUAGE,
    )
