"""
Speaker embedding computation utilities for VECL-TTS.
"""

from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

from vecl.utils.downloader import download_s3_file


def _get_speaker_manager(speaker_encoder_model_dir: Path):
    """
    Initializes the SpeakerManager using the pre-downloaded model files
    from the specified directory.
    """
    print('Initializing speaker manager...')
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


def compute_speaker_embeddings(
    dataset_configs: list[BaseDatasetConfig],
    embeddings_file_path: Path,
    speaker_encoder_model_dir: Path,
    s3_bucket: str,
    s3_key: str,
):
    """
    Computes speaker embeddings, attempting to download from S3 first.
    It also handles remapping from an old format if necessary.
    """
    if not embeddings_file_path.exists():
        print(
            'Speaker embeddings not found locally. Attempting to download from S3...'
        )
        download_s3_file(
            bucket_name=s3_bucket,
            s3_key=s3_key,
            local_path=embeddings_file_path,
        )

    if embeddings_file_path.exists():
        print(
            f'Speaker embeddings found at: {embeddings_file_path}. Validating format...'
        )
        try:
            embeddings = torch.load(
                embeddings_file_path, map_location=torch.device('cpu')
            )
            if embeddings and '#' in next(iter(embeddings.keys()), ''):
                print('✔️ Speaker embeddings file is in the correct format.')
                return
            else:
                print(
                    '⚠️ Speaker embeddings file is in an outdated format. Remapping...'
                )
                all_samples, _ = load_tts_samples(
                    dataset_configs, eval_split=False
                )
                audio_to_embedding = {}
                for sample in tqdm(all_samples, desc='Remapping embeddings'):
                    audio_unique_name = sample['audio_unique_name']
                    speaker_name = sample['speaker_name']
                    if speaker_name in embeddings:
                        audio_to_embedding[audio_unique_name] = embeddings[
                            speaker_name
                        ]
                torch.save(audio_to_embedding, embeddings_file_path)
                print(
                    f'✔️ Embeddings remapped and saved to: {embeddings_file_path}'
                )
                return
        except Exception as e:
            print(
                f'⚠️ Could not process or remap embeddings file ({e}). Re-computing.'
            )

    # Fallback to computation
    speaker_manager = _get_speaker_manager(speaker_encoder_model_dir)

    all_samples, _ = load_tts_samples(dataset_configs, eval_split=False)
    speaker_to_files = defaultdict(list)
    for sample in all_samples:
        speaker_to_files[sample['speaker_name']].append(sample['audio_file'])

    speaker_embeddings = {}
    for speaker_name, audio_files in tqdm(
        speaker_to_files.items(), desc='Computing Embeddings'
    ):
        try:
            embedding = speaker_manager.compute_embedding_from_clip(
                audio_files
            )
            speaker_embeddings[speaker_name] = {
                'name': speaker_name,
                'embedding': embedding,
            }
        except Exception as e:
            print(f'Failed to process {speaker_name}: {e}')

    audio_to_embedding = {}
    for sample in tqdm(all_samples, desc='Remapping embeddings'):
        audio_unique_name = sample['audio_unique_name']
        speaker_name = sample['speaker_name']
        if speaker_name in speaker_embeddings:
            audio_to_embedding[audio_unique_name] = speaker_embeddings[
                speaker_name
            ]

    torch.save(audio_to_embedding, embeddings_file_path)
    print(f'Embeddings saved to: {embeddings_file_path}')
