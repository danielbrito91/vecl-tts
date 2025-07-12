import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from tests.fixtures.data import SAMPLE_TTS_SAMPLES as samples
from vecl.embeddings.speaker import compute_speaker_embeddings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, duration_sec: float = 0.2, sr: int = 16000):
    """Write a silent WAV file via stdlib *only* (no external deps)."""
    n_samples = int(duration_sec * sr)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        silence = (0).to_bytes(2, byteorder='little', signed=True)
        wf.writeframes(silence * n_samples)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@patch('vecl.embeddings.speaker._get_speaker_manager')
@patch('vecl.embeddings.speaker.load_tts_samples')
@patch('vecl.embeddings.speaker.download_s3_file')
def test_compute_speaker_embeddings_integration(
    mock_download,
    mock_load_samples,
    mock_get_manager,
    temp_dataset_dir,
    mock_dataset_configs,
):
    """End-to-end test with tiny real WAV files and mocked SpeakerManager."""

    # ------------------------------------------------------------------
    # Set up tiny audio files
    # ------------------------------------------------------------------
    audio_dir = temp_dataset_dir / 'audio'
    audio_dir.mkdir()
    wav1 = audio_dir / 'audio_001.wav'
    wav2 = audio_dir / 'audio_002.wav'
    _write_wav(wav1)
    _write_wav(wav2)

    mock_load_samples.return_value = (samples, None)

    # Mock SpeakerManager
    manager = MagicMock()
    manager.compute_embedding_from_clip.return_value = torch.randn(512)
    mock_get_manager.return_value = manager

    embeddings_path = temp_dataset_dir / 'speaker_embeddings.pth'
    speaker_encoder_dir = temp_dataset_dir / 'speaker_encoder'
    speaker_encoder_dir.mkdir()

    compute_speaker_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_path,
        speaker_encoder_model_dir=speaker_encoder_dir,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    # Verify file written and contains correct keys
    emb = torch.load(embeddings_path, map_location='cpu')
    expected_keys = {s['audio_unique_name'] for s in samples}
    assert set(emb.keys()) == expected_keys
    assert manager.compute_embedding_from_clip.call_count == len({
        s['speaker_name'] for s in samples
    })
