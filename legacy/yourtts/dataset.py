import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.download import download_url

from vecl.utils.downloader import download_from_s3, extract_tar_file

SPEAKER_ENCODER_CHECKPOINT_URL = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar'
SPEAKER_ENCODER_CONFIG_URL = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json'

S3_BUCKET_NAME = 'hotmart-datascience-sagemaker'
S3_DATA_KEY = 'tts/cml-tts/processed_24k.tar.gz'


def prepare_dataset_configs(main_metadata_path: Path, dataset_base_path: Path):
    """
    Reads the main manifest, splits it by language, and creates a list of
    BaseDatasetConfig objects required for the Vits model.
    """
    print('>>> Preparing dataset configs from main manifest...')
    if not main_metadata_path.exists():
        print(f'Main metadata file not found at {main_metadata_path}')
        print('Attempting to download from S3...')
        try:
            local_tar_path = dataset_base_path / 'processed_data.tar.gz'
            download_from_s3(S3_BUCKET_NAME, S3_DATA_KEY, local_tar_path)
            print('  > ✅ Download complete.')

            print(f'Extracting {local_tar_path} to {dataset_base_path} ...')
            extract_tar_file(local_tar_path, dataset_base_path)

            extracted_dir_name = Path(S3_DATA_KEY).name.replace('.tar.gz', '')
            parent_dir = dataset_base_path / extracted_dir_name

            if parent_dir.is_dir():
                print(
                    f'  > Moving files from {parent_dir} to {dataset_base_path}'
                )
                for item_path in parent_dir.iterdir():
                    dest_path = dataset_base_path / item_path.name
                    shutil.move(str(item_path), str(dest_path))
                parent_dir.rmdir()
                print(
                    f'  > ✅ Files moved and {parent_dir.name} directory removed.'
                )

            assert main_metadata_path.exists(), (
                f'Main metadata file not found at {main_metadata_path}'
            )
        except Exception as e:
            print(f'  > ❌ Error downloading from S3: {e}')
            raise
    else:
        print(f'  > ✅ Main metadata file found at {main_metadata_path}')

    main_df = pd.read_csv(main_metadata_path, sep='|')
    main_df.columns = main_df.columns.str.strip()

    for col in main_df.select_dtypes(include=['object']).columns:
        main_df[col] = main_df[col].str.strip()

    unique_languages = main_df['language'].dropna().unique()
    print(f'Languages found: {unique_languages}')

    dataset_configs = []
    for lang in unique_languages:
        print(f'Processing language: {lang}...')
        lang_df = main_df[main_df['language'] == lang].copy()
        temp_meta_path = dataset_base_path / f'metadata_{lang}.csv'

        lang_df['speaker_name_combined'] = lang_df.apply(
            lambda row: f'{row["dataset"]}_{row["speaker_code"]}'
            + (
                f'_{row["speaker_gender"]}'
                if pd.notna(row.get('speaker_gender'))
                else ''
            ),
            axis=1,
        )

        output_df = pd.DataFrame({
            'full_path': 'audio/' + lang_df['filename'],
            'ignored': 'ignore',
            'text': lang_df['normalized_transcription'],
            'speaker_id': lang_df['speaker_name_combined'],
        })

        output_df.to_csv(temp_meta_path, sep='|', header=False, index=False)
        print(
            f"    ✅ Temporary manifest for '{lang}' created at: {temp_meta_path}"
        )

        dataset_configs.append(
            BaseDatasetConfig(
                formatter='brspeech',
                dataset_name=f'multilingual_custom_{lang}',
                meta_file_train=str(temp_meta_path.name),
                path=str(dataset_base_path),
                language=lang,
            )
        )
    return dataset_configs


def compute_speaker_embeddings(
    dataset_configs_list, embeddings_file_path, speaker_encoder_model_dir: Path
):
    """
    Computes speaker embeddings sequentially on the GPU to avoid memory issues,
    saving the result to a .pth file.
    """
    if embeddings_file_path.exists():
        print(
            f'>>> Found speaker embeddings file at: {embeddings_file_path}. Checking format...'
        )
        try:
            embeddings = torch.load(
                embeddings_file_path, map_location=torch.device('cpu')
            )
            if embeddings and '#' in next(iter(embeddings.keys()), ''):
                print('✅ Speaker embeddings file is in the correct format.')
                return
            else:
                print(
                    '⚠️ Speaker embeddings file is in an outdated format. Remapping to the correct format...'
                )
                all_samples, _ = load_tts_samples(
                    dataset_configs_list, eval_split=False
                )

                print(
                    '    > Remapping speaker embeddings to audio file names...'
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
                    f'\n✅ Embeddings for {len(audio_to_embedding)} audio files remapped and saved to: {embeddings_file_path}'
                )
                return  # Return after successful remapping
        except Exception as e:
            print(
                f'⚠️ Could not process or remap embeddings file ({e}). Proceeding to re-compute.'
            )

    # Check S3 only if the local file doesn't exist.
    try:
        import boto3

        s3_client = boto3.client('s3')
        speaker_embedding_key = 'tts/yourtts/embeddings/speakers.pth'

        try:
            s3_client.head_object(
                Bucket=S3_BUCKET_NAME, Key=speaker_embedding_key
            )
            print('✅ Speaker embeddings file found in S3')
            print('Downloading from S3...')
            download_from_s3(
                S3_BUCKET_NAME, speaker_embedding_key, embeddings_file_path
            )
            print(f'✅ Downloaded embeddings to: {embeddings_file_path}')
            # After downloading, recursively call to validate and remap if necessary
            print('>>> Validating newly downloaded embeddings file...')
            compute_speaker_embeddings(
                dataset_configs_list,
                embeddings_file_path,
                speaker_encoder_model_dir,
            )
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(
                    '❌ Speaker embeddings not found in S3, proceeding with computation...'
                )
            else:
                raise
    except Exception as e:
        print(f'❌ Error checking S3: {e}')

    print('\n>>> Computing speaker embeddings (Sequential GPU Strategy)...')
    speaker_encoder_model_dir.mkdir(exist_ok=True)
    se_checkpoint_path = speaker_encoder_model_dir / 'model_se.pth.tar'
    se_config_path = speaker_encoder_model_dir / 'config_se.json'

    if not se_checkpoint_path.exists():
        print(
            f'Downloading speaker encoder checkpoint from {SPEAKER_ENCODER_CHECKPOINT_URL} to {se_checkpoint_path}'
        )
        download_url(SPEAKER_ENCODER_CHECKPOINT_URL, str(se_checkpoint_path))
    if not se_config_path.exists():
        print(
            f'Downloading speaker encoder config from {SPEAKER_ENCODER_CONFIG_URL} to {se_config_path}'
        )
        download_url(SPEAKER_ENCODER_CONFIG_URL, str(se_config_path))

    print('    > Loading Speaker Encoder on GPU...')
    speaker_manager = SpeakerManager(use_cuda=torch.cuda.is_available())
    speaker_manager.init_encoder(str(se_checkpoint_path), str(se_config_path))
    print('    ✅ Speaker Encoder loaded.')

    print('    > Grouping audio files by speaker...')
    all_samples, _ = load_tts_samples(dataset_configs_list, eval_split=False)
    speaker_to_files = defaultdict(list)
    for sample in all_samples:
        speaker_to_files[sample['speaker_name']].append(sample['audio_file'])

    speaker_embeddings = {}
    for speaker_name, audio_files in tqdm(
        speaker_to_files.items(), desc='Computing Embeddings (GPU)'
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
            print(f'\n    [ERROR] Failed to process {speaker_name}: {e}')

    print('    > Remapping speaker embeddings to audio file names...')
    audio_to_embedding = {}
    for sample in tqdm(all_samples, desc='Remapping embeddings'):
        audio_unique_name = sample['audio_unique_name']
        speaker_name = sample['speaker_name']
        if speaker_name in speaker_embeddings:
            audio_to_embedding[audio_unique_name] = speaker_embeddings[
                speaker_name
            ]

    torch.save(audio_to_embedding, embeddings_file_path)
    print(
        f'\n✅ Embeddings for {len(audio_to_embedding)} audio files saved to: {embeddings_file_path}'
    )
