import logging
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

from vecl.utils.downloader import download_s3_file

logger = logging.getLogger(__name__)


def compute_speaker_embeddings(
    dataset_configs: list[BaseDatasetConfig],
    embeddings_file_path: Path,
    speaker_encoder_model_dir: Path,
    s3_bucket: str,
    s3_key: str,
):
    """
    Ensure `embeddings_file_path` exists in a single, canonical format
    (keyed by ``audio_unique_name`` → embedding dict). If the file is
    already present locally **or** after an optional S3 download, the
    function returns early. Otherwise it computes the embeddings from
    scratch and writes the file.
    """
    if not embeddings_file_path.exists():
        logger.info(
            'Speaker embeddings not found locally. Attempting to download from S3...'
        )
        download_success = download_s3_file(
            bucket_name=s3_bucket,
            s3_key=s3_key,
            local_path=embeddings_file_path,
        )
        if not download_success:
            logger.warning('Failed to download embeddings from S3.')

    if _embeddings_cover_dataset(embeddings_file_path, dataset_configs):
        logger.info('✔️ Existing embeddings cover all audio clips.')
        return

    # Fallback – compute from scratch
    speaker_manager = _get_speaker_manager(speaker_encoder_model_dir)

    all_samples, _ = load_tts_samples(dataset_configs, eval_split=False)
    if not all_samples:
        logger.warning('No samples found – saving empty embeddings file.')
        torch.save({}, embeddings_file_path)
        return

    speaker_embs = _compute_embeddings_per_speaker(
        all_samples, speaker_manager
    )
    audio_embs = _remap_speaker_to_audio_embeddings(speaker_embs, all_samples)

    torch.save(audio_embs, embeddings_file_path)
    logger.info(f'Embeddings saved to: {embeddings_file_path}')


def _embeddings_cover_dataset(
    embeddings_file_path: Path, dataset_configs: list[BaseDatasetConfig]
) -> bool:
    """Return ``True`` iff *all* dataset samples have an embedding entry.

    A *sample* is identified by its ``audio_unique_name`` key. If the file
    can't be loaded or some keys are missing, the function returns ``False``.
    """

    if not embeddings_file_path.is_file():
        return False

    try:
        embeddings: dict = torch.load(
            embeddings_file_path, map_location=torch.device('cpu')
        )
    except Exception as exc:
        logger.error(f'⚠️ Failed to load embeddings file: {exc}')
        return False

    # Early-out if dataset is empty – nothing to validate.
    all_samples, _ = load_tts_samples(dataset_configs, eval_split=False)
    if not all_samples:
        return True

    required_keys = {s['audio_unique_name'] for s in all_samples}
    return required_keys.issubset(embeddings.keys())


def _compute_embeddings_per_speaker(
    all_samples: list[dict], speaker_manager: SpeakerManager
) -> dict[str, dict]:
    """Compute an embedding per speaker.

    Parameters
    ----------
    all_samples
        List of dataset sample dictionaries produced by ``load_tts_samples``.
    speaker_manager
        Pre-initialised ``SpeakerManager`` with an encoder.

    Returns
    -------
    dict
        Mapping ``speaker_name → {name, embedding}``.
    """

    speaker_to_files: dict[str, list[str]] = defaultdict(list)
    for sample in all_samples:
        speaker_to_files[sample['speaker_name']].append(sample['audio_file'])

    speaker_embeddings: dict[str, dict] = {}
    for speaker_name, audio_files in tqdm(
        speaker_to_files.items(), desc='Computing speaker embeddings'
    ):
        try:
            embedding = speaker_manager.compute_embedding_from_clip(
                audio_files
            )
            speaker_embeddings[speaker_name] = {
                'name': speaker_name,
                'embedding': embedding,
            }
        except Exception as exc:
            logger.error(
                f'⚠️ Failed to compute embedding for {speaker_name}: {exc}'
            )

    return speaker_embeddings


def _remap_speaker_to_audio_embeddings(
    speaker_embeddings: dict[str, dict], all_samples: list[dict]
) -> dict[str, dict]:
    """Convert *speaker*-keyed dict into *audio*-keyed dict."""

    audio_to_embedding: dict[str, dict] = {}
    for sample in all_samples:
        spk = sample['speaker_name']
        if spk in speaker_embeddings:
            audio_to_embedding[sample['audio_unique_name']] = (
                speaker_embeddings[spk]
            )
    return audio_to_embedding


def _get_speaker_manager(speaker_encoder_model_dir: Path):
    """
    Initializes the SpeakerManager using the pre-downloaded model files
    from the specified directory.
    """
    logger.info('Initializing speaker manager...')
    se_checkpoint_path = speaker_encoder_model_dir / 'model_se.pth.tar'
    se_config_path = speaker_encoder_model_dir / 'config_se.json'

    if not se_checkpoint_path.is_file() or not se_config_path.is_file():
        raise FileNotFoundError(
            f'Speaker encoder model/config not found in {speaker_encoder_model_dir}. '
            'Ensure they are downloaded before running this step.'
        )

    speaker_manager = SpeakerManager(use_cuda=torch.cuda.is_available())
    speaker_manager.init_encoder(str(se_checkpoint_path), str(se_config_path))
    return speaker_manager
