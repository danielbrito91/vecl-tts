from unittest.mock import MagicMock, patch

import pytest
import torch

from vecl.embeddings.speaker import (
    _compute_embeddings_per_speaker,
    _embeddings_cover_dataset,
    _get_speaker_manager,
    _remap_speaker_to_audio_embeddings,
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


@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_existing_file_correct_format(
    mock_download,
    mock_get_manager,
    mock_load_samples,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Early-exit when embeddings file already contains all required keys."""
    embeddings_file = temp_dataset_dir / 'speaker_embeddings.pth'

    # Create embeddings dict with **all** sample keys
    torch.save(
        {
            s['audio_unique_name']: {'embedding': [1.0]}
            for s in sample_tts_samples
        },
        embeddings_file,
    )

    # Mock dataset samples used for validation
    mock_load_samples.return_value = (sample_tts_samples, None)

    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    mock_download.assert_not_called()
    mock_get_manager.assert_not_called()
    mock_load_samples.assert_called_once()


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
    assert mock_speaker_manager.compute_embedding_from_clip.call_count == 5
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
        if any('coraaa-486-00000' in f for f in audio_files):
            raise Exception('Failed to process speaker')
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
    assert len(audio_to_embedding) == 4
    assert (
        'multilingual_custom_pt-br#audio/coraaa-904-00001'
        in audio_to_embedding
    )
    assert (
        'multilingual_custom_pt-br#audio/coraaa-486-00000'
        not in audio_to_embedding
    )


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


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


@patch('vecl.embeddings.speaker.load_tts_samples')
def test_embeddings_cover_dataset_true(
    mock_load_samples,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Returns True when file contains embeddings for every sample."""
    emb_path = temp_dataset_dir / 'embeddings.pth'
    torch.save(
        {s['audio_unique_name']: {'embedding': 0} for s in sample_tts_samples},
        emb_path,
    )

    mock_load_samples.return_value = (sample_tts_samples, None)

    assert _embeddings_cover_dataset(emb_path, mock_dataset_configs) is True


@patch('vecl.embeddings.speaker.load_tts_samples')
def test_embeddings_cover_dataset_false_missing_keys(
    mock_load_samples,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Returns False when at least one key is missing."""
    emb_path = temp_dataset_dir / 'embeddings.pth'
    # Only first sample key present
    torch.save(
        {sample_tts_samples[0]['audio_unique_name']: {'embedding': 0}},
        emb_path,
    )

    mock_load_samples.return_value = (sample_tts_samples, None)

    assert _embeddings_cover_dataset(emb_path, mock_dataset_configs) is False


def test_compute_embeddings_per_speaker_calls_encoder(sample_tts_samples):
    """Ensure one encoder call per unique speaker."""
    mock_sm = MagicMock()
    mock_sm.compute_embedding_from_clip.return_value = torch.randn(512)

    speaker_embs = _compute_embeddings_per_speaker(sample_tts_samples, mock_sm)

    # Two unique speakers in fixture
    assert len(speaker_embs) == 5
    assert mock_sm.compute_embedding_from_clip.call_count == 5


def test_remap_speaker_to_audio_embeddings(sample_tts_samples):
    """All audio keys should be present after remap."""
    speaker_embs = {
        s['speaker_name']: {'name': s['speaker_name'], 'embedding': i}
        for i, s in enumerate(
            sorted(
                sample_tts_samples,
                key=lambda x: x['speaker_name'],
                reverse=True,
            )
        )
    }

    audio_embs = _remap_speaker_to_audio_embeddings(
        speaker_embs, sample_tts_samples
    )

    expected_keys = {s['audio_unique_name'] for s in sample_tts_samples}
    assert set(audio_embs.keys()) == expected_keys
