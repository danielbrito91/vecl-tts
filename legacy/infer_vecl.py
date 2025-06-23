import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from TTS.config import load_config as coqui_load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from vecl.vecl.emotion_embedding import EmotionEmbedding

# Prefer the custom VECL model (with emotion projection) when available
try:
    from vecl.vecl.vecl import Vecl  # noqa: E402
except ImportError:
    Vecl = None


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def patch_state_dict(sd: dict) -> dict:
    """Convert weight_norm `weight_g/weight_v` keys to the newer format"""
    patched = {}
    for k, v in sd.items():
        if '.weight_g' in k:
            k = k.replace('.weight_g', '.parametrizations.weight.original0')
        elif '.weight_v' in k:
            k = k.replace('.weight_v', '.parametrizations.weight.original1')
        patched[k] = v
    return patched


def save_wav(path: Path, audio: np.ndarray, sr: int = 24_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, 'PCM_16')


def load_model_and_cfg(
    checkpoint: Path,
    config_json: Path,
    language_ids: Path | None,
    device: torch.device,
):
    # ------------------------------------------------------------------ #
    # 1. Load & massage config                                           #
    # ------------------------------------------------------------------ #
    cfg = coqui_load_config(config_json)
    # some older configs lack audio section when loaded via coqpit
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

    # turn off extras not needed at inference
    cfg.model_args.d_vector_file = None
    cfg.model_args.use_speaker_encoder_as_loss = False

    # ------------------------------------------------------------------ #
    # Ensure language embeddings path is valid                           #
    # ------------------------------------------------------------------ #
    # Some training configs keep an absolute path (e.g. SageMaker) that
    # is not available at inference time.  Override it with the path
    # provided by the caller (``language_ids``) or disable it entirely.
    if language_ids is not None:
        cfg.language_ids_file = str(language_ids)
        cfg.model_args.language_ids_file = str(language_ids)
    else:
        # Disable language embeddings altogether if none is given.
        cfg.language_ids_file = None
        cfg.model_args.language_ids_file = None

    # ------------------------------------------------------------------ #
    # Fix speaker-encoder paths                                         #
    # ------------------------------------------------------------------ #
    # Training configs often keep absolute SageMaker paths that are not
    # valid on the local machine.  Replace them with local copies if the
    # files are missing.
    from pathlib import Path as _P  # local import to avoid polluting namespace

    se_model_path = _P(str(cfg.model_args.speaker_encoder_model_path))
    se_cfg_path = _P(str(cfg.model_args.speaker_encoder_config_path))

    # if either file is missing, attempt to locate a local fallback
    if not se_model_path.is_file() or not se_cfg_path.is_file():
        fallback_models = [
            _P(str(DEFAULT_SE_CHECKPOINT)),
            _P(str(ALT_SE_CHECKPOINT)),
        ]
        fallback_cfgs = [_P(str(DEFAULT_SE_CONFIG)), _P(str(ALT_SE_CONFIG))]

        for fb in fallback_models:
            if fb.is_file():
                se_model_path = fb
                break

        for fb in fallback_cfgs:
            if fb.is_file():
                se_cfg_path = fb
                break

        cfg.model_args.speaker_encoder_model_path = str(se_model_path)
        cfg.model_args.speaker_encoder_config_path = str(se_cfg_path)

    # ------------------------------------------------------------------ #
    # 2. Build model shell – use custom Vecl if available and emotion support
    # ------------------------------------------------------------------ #
    if Vecl is not None and hasattr(cfg.model_args, 'emotion_embedding_dim'):
        model = Vecl.init_from_config(cfg)
    else:
        model = setup_model(cfg)

    # ------------------------- Load checkpoint ------------------------ #
    sd = torch.load(checkpoint, map_location='cpu')['model'].copy()

    # Remove speaker-encoder parameters (not required at inference)
    for key in list(sd.keys()):
        if 'speaker_encoder' in key:
            del sd[key]

    # If the checkpoint was fine-tuned on a different set of languages the
    # language-embedding matrix might have a different size.  Skip loading
    # it if dimensions don't match the current model.
    if 'emb_l.weight' in sd:
        pre_shape = sd['emb_l.weight'].shape
        cur_shape = model.emb_l.weight.shape
        if pre_shape != cur_shape:
            print(
                f'⚠️  Language embedding mismatch {pre_shape} vs {cur_shape}; '
                'ignoring checkpoint embeddings.'
            )
            del sd['emb_l.weight']

    # If the model unexpectedly lacks the emotion projection but the checkpoint
    # contains it, create it on-the-fly so weights load without warnings.
    if (
        not hasattr(model, 'emotion_proj') or model.emotion_proj is None
    ) and 'emotion_proj.proj.weight' in sd:
        w_shape = sd['emotion_proj.proj.weight'].shape  # (out_dim, in_dim)
        in_dim = w_shape[1]
        out_dim = w_shape[0]
        from vecl.vecl.emotion_embedding import EmotionProj  # local import

        model.emotion_proj = EmotionProj(in_dim, out_dim)
        # register in parent so state-dict key matches
        setattr(model, 'emotion_proj', model.emotion_proj)

    # Patch keys for newer weight-norm format and load
    sd = patch_state_dict(sd)
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys:
        print('ℹ️ Missing keys:', load_res.missing_keys)
    if load_res.unexpected_keys:
        print('ℹ️ Unexpected keys:', load_res.unexpected_keys)

    # ------------------------------------------------------------------ #
    # 3. Language & speaker helpers                                      #
    # ------------------------------------------------------------------ #
    lm = LanguageManager(language_ids) if language_ids else None
    model.language_manager = lm

    model.eval().to(device)
    return model, cfg


def compute_ref_embeddings(
    spk_mgr: SpeakerManager, ref_wav: Path, device: torch.device
):
    dvec = spk_mgr.compute_embedding_from_clip(ref_wav)
    dvec = torch.FloatTensor(dvec).to(device)

    emo_emb = EmotionEmbedding().get_emotion_embedding(str(ref_wav))
    emo_emb = emo_emb.to(device).squeeze(0)  # 1024

    return dvec, emo_emb


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser('VECL-TTS inference')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--language_ids', type=Path, help='language_ids.json')
    p.add_argument('--text', required=True)
    p.add_argument('--language', default='pt-br')
    p.add_argument('--reference_wav', type=Path)
    p.add_argument('--out_wav', type=Path, default=Path('vecl_out.wav'))
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--noise_scale', type=float, default=0.3)
    p.add_argument('--noise_scale_dp', type=float, default=0.3)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------------
    model, cfg = load_model_and_cfg(
        args.checkpoint, args.config, args.language_ids, device
    )

    # Speaker manager for d-vector extraction (if reference provided)
    spk_mgr = SpeakerManager(
        encoder_model_path=cfg.model_args.speaker_encoder_model_path,
        encoder_config_path=cfg.model_args.speaker_encoder_config_path,
        use_cuda=device.type == 'cuda',
    )

    # ----------------------------------------------------------------------
    # Prepare auxiliary inputs
    # ----------------------------------------------------------------------
    aux = {'d_vectors': None, 'language_ids': None, 'emotion_embeddings': None}

    if args.reference_wav:
        dvec, emo = compute_ref_embeddings(spk_mgr, args.reference_wav, device)
        aux['d_vectors'] = dvec.unsqueeze(0)
        aux['emotion_embeddings'] = emo.unsqueeze(0)

    if (
        model.language_manager
        and args.language in model.language_manager.name_to_id
    ):
        lang_id = model.language_manager.name_to_id[args.language]
        aux['language_ids'] = torch.LongTensor([lang_id]).to(device)
    else:
        print(
            f"⚠️ language '{args.language}' not in checkpoint; defaulting to 0"
        )

    # ----------------------------------------------------------------------
    # Synthesis settings (mirror infer_02.py)
    # ----------------------------------------------------------------------
    model.length_scale = 2.0 - args.speed
    model.inference_noise_scale = args.noise_scale
    model.inference_noise_scale_dp = args.noise_scale_dp

    # ----------------------------------------------------------------------
    # Tokenise & run inference
    # ----------------------------------------------------------------------

    lang_id = None
    if (
        model.language_manager
        and args.language in model.language_manager.name_to_id
    ):
        lang_id = model.language_manager.name_to_id[args.language]

    with torch.no_grad():
        wav, *_ = synthesis(
            model=model,
            text=args.text,
            CONFIG=cfg,
            use_cuda=device.type == 'cuda',
            speaker_id=None,
            style_wav=None,
            style_text=None,
            d_vector=aux['d_vectors'].squeeze(0)
            if aux['d_vectors'] is not None
            else None,
            language_id=lang_id,
            use_griffin_lim=False,
        ).values()

    save_wav(args.out_wav, wav, sr=cfg.audio.sample_rate)
    print(f'✅ Saved to {args.out_wav}')


# -----------------------------------------------------------------------------
# Public helper – easier programmatic use (mirrors yourtts/infer_02.generate)
# -----------------------------------------------------------------------------


def generate(
    checkpoint: Path,
    config_json: Path,
    language_ids_json: Path | None,
    sentence: str,
    ref_wav: Path | None,
    target_language: str,
    out_wav: Path,
    speed: float = 1.0,
    noise_scale: float = 0.3,
    noise_scale_dp: float = 0.3,
):
    """High-level TTS synthesis wrapper.

    Parameters
    ----------
    checkpoint / config_json : files from a VECL or CML run directory.
    language_ids_json : language_ids.json (or None).
    sentence : text to speak.
    ref_wav : wav file used to extract speaker + emotion.  If None, model defaults.
    target_language : code present in language_ids.json (e.g. 'pt-br').
    out_wav : where to save the resulting waveform.
    speed : 1.0 = normal (length_scale = 1.0).  Higher → faster.
    noise_scale / noise_scale_dp : VITS noise parameters.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, cfg = load_model_and_cfg(
        checkpoint, config_json, language_ids_json, device
    )

    # Speaker manager
    spk_mgr = SpeakerManager(
        encoder_model_path=cfg.model_args.speaker_encoder_model_path,
        encoder_config_path=cfg.model_args.speaker_encoder_config_path,
        use_cuda=device.type == 'cuda',
    )

    aux = {'d_vectors': None, 'language_ids': None, 'emotion_embeddings': None}

    if ref_wav is not None:
        dvec, emo = compute_ref_embeddings(spk_mgr, ref_wav, device)
        aux['d_vectors'] = dvec.unsqueeze(0)
        aux['emotion_embeddings'] = emo.unsqueeze(0)

    if (
        model.language_manager
        and target_language in model.language_manager.name_to_id
    ):
        lang_id = model.language_manager.name_to_id[target_language]
        aux['language_ids'] = torch.LongTensor([lang_id]).to(device)

    # Inference settings
    model.length_scale = 2.0 - speed
    model.inference_noise_scale = noise_scale
    model.inference_noise_scale_dp = noise_scale_dp

    with torch.no_grad():
        wav, *_ = synthesis(
            model=model,
            text=sentence,
            CONFIG=cfg,
            use_cuda=device.type == 'cuda',
            speaker_id=None,
            style_wav=None,
            style_text=None,
            d_vector=aux['d_vectors'].squeeze(0)
            if aux['d_vectors'] is not None
            else None,
            language_id=lang_id,
            use_griffin_lim=False,
        ).values()

    save_wav(out_wav, wav, sr=cfg.audio.sample_rate)
    return out_wav


# -----------------------------------------------------------------------------
# Default LOCAL paths for the speaker-encoder
# These mirror the constants defined in `vecl/yourtts/infer_02.py` so that both
# inference helpers resolve the same checkpoint/config without depending on
# absolute SageMaker paths.
# -----------------------------------------------------------------------------

DEFAULT_SE_CHECKPOINT = Path(
    'models/checkpoints_yourtts_cml_tts_dataset/model_se.pth'
)
DEFAULT_SE_CONFIG = Path(
    'models/checkpoints_yourtts_cml_tts_dataset/config_se.json'
)

# A second set of fallbacks (coqui's original tar format)
ALT_SE_CHECKPOINT = Path('speaker_encoder_model/model_se.pth.tar')
ALT_SE_CONFIG = Path('speaker_encoder_model/config.json')

if __name__ == '__main__':
    main()
