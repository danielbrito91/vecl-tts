"""
Test cases for speaker embedding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vecl.embeddings.speaker import (
    _get_speaker_manager,
    compute_speaker_embeddings,
)


@patch('vecl.embeddings.speaker.SpeakerManager')
def test_get_speaker_manager_success(mock_speaker_manager, temp_dataset_dir):
    """Test that SpeakerManager is initialized when model files exist."""
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'
    speaker_encoder_dir.mkdir()
    (speaker_encoder_dir / 'model_se.pth.tar').touch()
    (speaker_encoder_dir / 'config_se.json').touch()

    _get_speaker_manager(speaker_encoder_dir)

    mock_speaker_manager.assert_called_once_with(
        use_cuda=torch.cuda.is_available()
    )
    mock_speaker_manager.return_value.init_encoder.assert_called_once()


def test_get_speaker_manager_files_missing(temp_dataset_dir):
    """Test that FileNotFoundError is raised when model files are missing."""
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'
    speaker_encoder_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        _get_speaker_manager(speaker_encoder_dir)


@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_existing_file_correct_format(
    mock_download, mock_get_manager, temp_dataset_dir, mock_dataset_configs
):
    """Test that the function returns early if a valid embeddings file exists."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'
    # Create a dummy embeddings file in the correct format (with '#')
    torch.save(
        {'audio_001#speaker_001': {'embedding': [1.0]}}, embeddings_file
    )
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )
    # Neither download nor computation should be called
    mock_download.assert_not_called()
    mock_get_manager.assert_not_called()


@patch('vecl.embeddings.speaker.torch.save')
@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
@patch('vecl.embeddings.speaker.torch.load')
def test_compute_speaker_embeddings_existing_file_remap(
    mock_torch_load,
    mock_download,
    mock_get_manager,
    mock_load_samples,
    mock_torch_save,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test that an old-format embeddings file is remapped."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    # Old format embeddings (keyed by speaker name)
    old_format_embeddings = {
        'speaker_001': {'name': 'speaker_001', 'embedding': torch.randn(512)},
        'speaker_002': {'name': 'speaker_002', 'embedding': torch.randn(512)},
    }
    mock_torch_load.return_value = old_format_embeddings
    mock_load_samples.return_value = (sample_tts_samples, None)
    # Make the file exist
    embeddings_file.touch()

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )
    # No download should be attempted since file exists
    mock_download.assert_not_called()
    # No computation should be run
    mock_get_manager.assert_not_called()
    # Should load samples for remapping and save the new format
    mock_load_samples.assert_called_once()
    mock_torch_save.assert_called_once()
    # Check that the saved output has the correct format
    saved_data = mock_torch_save.call_args[0][0]
    assert 'audio_001#speaker_001' in saved_data


@patch('vecl.embeddings.speaker.torch.save')
@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_success(
    mock_download,
    mock_get_manager,
    mock_load_samples,
    mock_torch_save,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test successful speaker embeddings computation."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    mock_load_samples.return_value = (sample_tts_samples, None)
    mock_speaker_manager = MagicMock()
    mock_get_manager.return_value = mock_speaker_manager
    mock_speaker_manager.compute_embedding_from_clip.return_value = (
        torch.randn(512)
    )

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    mock_download.assert_called_once()
    mock_load_samples.assert_called_once()
    mock_get_manager.assert_called_once_with(speaker_encoder_dir)
    assert mock_speaker_manager.compute_embedding_from_clip.call_count == 2
    mock_torch_save.assert_called_once()


@patch('vecl.embeddings.speaker.torch.save')
@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_with_errors(
    mock_download,
    mock_get_manager,
    mock_load_samples,
    mock_torch_save,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test speaker embeddings computation with some failures."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    mock_load_samples.return_value = (sample_tts_samples, None)
    mock_speaker_manager = MagicMock()
    mock_get_manager.return_value = mock_speaker_manager

    def mock_compute_embedding_with_error(audio_files):
        if any('audio_001' in f or 'audio_003' in f for f in audio_files):
            raise Exception('Failed to process speaker_001')
        return torch.randn(512)

    mock_speaker_manager.compute_embedding_from_clip.side_effect = (
        mock_compute_embedding_with_error
    )

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args[0]
    audio_to_embedding = call_args[0]
    assert len(audio_to_embedding) == 1
    assert 'audio_002#speaker_002' in audio_to_embedding
    assert 'audio_001#speaker_001' not in audio_to_embedding


@patch('vecl.embeddings.speaker.torch.save')
@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_empty_samples(
    mock_download,
    mock_get_manager,
    mock_load_samples,
    mock_torch_save,
    temp_dataset_dir,
    mock_dataset_configs,
):
    """Test speaker embeddings computation with no samples."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    mock_load_samples.return_value = ([], None)
    mock_get_manager.return_value = MagicMock()

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args[0]
    audio_to_embedding = call_args[0]
    assert len(audio_to_embedding) == 0
