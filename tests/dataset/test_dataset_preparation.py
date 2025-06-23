"""
Test cases for dataset preparation functionality.
"""

import shutil
from unittest.mock import patch

import pandas as pd
import pytest
from TTS.config.shared_configs import BaseDatasetConfig

from vecl.dataset.preparation import prepare_dataset_configs


def test_prepare_configs_with_existing_metadata(
    temp_dataset_dir, sample_metadata_df, create_metadata_file
):
    """Test dataset config preparation when metadata file exists locally."""
    # Create metadata file
    create_metadata_file(temp_dataset_dir, sample_metadata_df)

    # Call the function
    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    # Verify results
    assert isinstance(configs, list)
    assert len(configs) == 2  # Should have configs for 'en' and 'pt-br'

    # Check that configs are BaseDatasetConfig objects
    for config in configs:
        assert isinstance(config, BaseDatasetConfig)
        assert config.language in ['en', 'pt-br']
        assert config.formatter == 'brspeech'
        assert config.dataset_name.startswith('multilingual_custom_')
        assert config.path == str(temp_dataset_dir)

    # Verify language-specific metadata files were created
    assert (temp_dataset_dir / 'metadata_en.csv').exists()
    assert (temp_dataset_dir / 'metadata_pt-br.csv').exists()

    # Verify content of language-specific files
    en_df = pd.read_csv(
        temp_dataset_dir / 'metadata_en.csv', sep='|', header=None
    )
    ptbr_df = pd.read_csv(
        temp_dataset_dir / 'metadata_pt-br.csv', sep='|', header=None
    )
    assert len(en_df) == 2  # 2 English samples
    assert len(ptbr_df) == 2  # 2 Portuguese samples


def test_speaker_name_combination(
    temp_dataset_dir, sample_metadata_df, create_metadata_file
):
    """Test that speaker names are correctly combined."""
    create_metadata_file(temp_dataset_dir, sample_metadata_df)

    prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    # Check one of the language-specific files for speaker naming
    en_metadata = temp_dataset_dir / 'metadata_en.csv'
    en_df = pd.read_csv(
        en_metadata,
        sep='|',
        header=None,
        names=['full_path', 'ignored', 'text', 'speaker_id'],
    )

    # Verify speaker IDs follow expected pattern: dataset_speakercode_gender
    expected_speakers = ['dataset1_sp001_M', 'dataset2_sp001_M']
    actual_speakers = en_df['speaker_id'].tolist()
    assert all(speaker in expected_speakers for speaker in actual_speakers)


def test_missing_metadata_no_s3(temp_dataset_dir):
    """Test error when metadata doesn't exist and no S3 bucket provided."""
    with pytest.raises(FileNotFoundError) as exc_info:
        prepare_dataset_configs(
            dataset_base_path=temp_dataset_dir, metadata_file='nonexistent.csv'
        )

    assert 'Metadata file not found' in str(exc_info.value)
    assert 'no S3 bucket provided' in str(exc_info.value)


@patch('vecl.dataset.preparation.download_from_s3')
@patch('vecl.dataset.preparation.extract_tar_file')
def test_s3_download_success(
    mock_extract,
    mock_download,
    temp_dataset_dir,
    sample_metadata_df,
    create_metadata_file,
):
    """Test successful S3 download when metadata doesn't exist locally."""
    # Setup mocks
    mock_download.return_value = None
    mock_extract.return_value = None

    # Create the metadata file in a subdirectory (simulating extraction)
    extracted_dir = temp_dataset_dir / 'processed_24k'
    extracted_dir.mkdir()
    create_metadata_file(extracted_dir, sample_metadata_df)

    # Mock the file movement that happens after extraction
    def mock_extract_side_effect(tar_path, extract_path):
        # Simulate moving files from extracted_dir to base_path
        metadata_source = extracted_dir / 'metadata.csv'
        metadata_dest = temp_dataset_dir / 'metadata.csv'
        if metadata_source.exists():
            shutil.move(str(metadata_source), str(metadata_dest))
        extracted_dir.rmdir()

    mock_extract.side_effect = mock_extract_side_effect

    # Call function
    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir,
        metadata_file='metadata.csv',
        s3_bucket_name='test-bucket',
    )

    # Verify S3 functions were called
    mock_download.assert_called_once()
    mock_extract.assert_called_once()

    # Verify configs were created
    assert len(configs) == 2


@patch('vecl.dataset.preparation.download_from_s3')
def test_s3_download_failure(mock_download, temp_dataset_dir):
    """Test handling of S3 download failure."""
    mock_download.side_effect = Exception('S3 download failed')

    with pytest.raises(Exception, match='S3 download failed'):
        prepare_dataset_configs(
            dataset_base_path=temp_dataset_dir,
            metadata_file='metadata.csv',
            s3_bucket_name='test-bucket',
        )


def test_single_language_dataset(temp_dataset_dir, create_metadata_file):
    """Test dataset with only one language."""
    single_lang_df = pd.DataFrame({
        'filename': ['audio_001.wav', 'audio_002.wav'],
        'language': ['en', 'en'],
        'normalized_transcription': ['hello one', 'hello two'],
        'dataset': ['test_dataset', 'test_dataset'],
        'speaker_code': ['sp001', 'sp002'],
        'speaker_gender': ['M', 'F'],
    })

    create_metadata_file(temp_dataset_dir, single_lang_df)

    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    assert len(configs) == 1
    assert configs[0].language == 'en'


def test_empty_metadata(temp_dataset_dir, create_metadata_file):
    """Test handling of empty metadata file."""
    empty_df = pd.DataFrame(
        columns=[
            'filename',
            'language',
            'normalized_transcription',
            'dataset',
            'speaker_code',
            'speaker_gender',
        ]
    )

    create_metadata_file(temp_dataset_dir, empty_df)

    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    assert configs == []


def test_missing_gender_column(temp_dataset_dir, create_metadata_file):
    """Test handling when speaker_gender column is missing."""
    df_no_gender = pd.DataFrame({
        'filename': ['audio_001.wav', 'audio_002.wav'],
        'language': ['en', 'pt-br'],
        'normalized_transcription': ['hello world', 'olá mundo'],
        'dataset': ['dataset1', 'dataset1'],
        'speaker_code': ['sp001', 'sp002'],
        # No speaker_gender column
    })

    create_metadata_file(temp_dataset_dir, df_no_gender)

    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    # Should still work, just without gender in speaker names
    assert len(configs) == 2

    # Check that speaker names don't have trailing underscores
    en_metadata = temp_dataset_dir / 'metadata_en.csv'
    en_df = pd.read_csv(
        en_metadata,
        sep='|',
        header=None,
        names=['full_path', 'ignored', 'text', 'speaker_id'],
    )
    speaker_id = en_df['speaker_id'].iloc[0]
    assert not speaker_id.endswith('_')


@patch('vecl.dataset.preparation.download_from_s3')
@patch('vecl.dataset.preparation.extract_tar_file')
def test_custom_s3_key(
    mock_extract,
    mock_download,
    temp_dataset_dir,
    sample_metadata_df,
    create_metadata_file,
):
    """Test using custom S3 data key."""

    # Setup mocks to create metadata file
    def setup_metadata(*args):
        create_metadata_file(temp_dataset_dir, sample_metadata_df)

    mock_extract.side_effect = setup_metadata

    custom_key = 'custom/path/data.tar.gz'
    prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir,
        metadata_file='metadata.csv',
        s3_bucket_name='test-bucket',
        s3_data_key=custom_key,
    )

    # Verify custom key was used
    call_args = mock_download.call_args
    assert (
        call_args[0][1] == custom_key
    )  # Second argument should be the S3 key


def test_whitespace_handling(temp_dataset_dir, create_metadata_file):
    """Test that whitespace in CSV data is properly stripped."""
    df_with_whitespace = pd.DataFrame({
        'filename': [' audio_001.wav ', ' audio_002.wav '],
        'language': [' en ', ' pt-br '],
        'normalized_transcription': [' hello world ', ' olá mundo '],
        'dataset': [' dataset1 ', ' dataset1 '],
        'speaker_code': [' sp001 ', ' sp002 '],
        'speaker_gender': [' M ', ' F '],
    })

    create_metadata_file(temp_dataset_dir, df_with_whitespace)

    configs = prepare_dataset_configs(
        dataset_base_path=temp_dataset_dir, metadata_file='metadata.csv'
    )

    # Check that language values were stripped properly
    languages = [config.language for config in configs]
    assert 'en' in languages
    assert 'pt-br' in languages
    assert ' en ' not in languages  # Should not have whitespace
