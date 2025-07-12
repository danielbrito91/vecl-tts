import os
from unittest.mock import MagicMock, patch

import torch

from vecl.embeddings.emotion import (
    EmotionEmbedding,
    compute_emotion_embeddings,
)

# Dummy model name for testing
DUMMY_SER_MODEL_NAME = 'dummy/ser-model'


@patch(
    'vecl.embeddings.emotion.AutoModelForAudioClassification.from_pretrained'
)
@patch('vecl.embeddings.emotion.AutoFeatureExtractor.from_pretrained')
@patch('vecl.embeddings.emotion.torch.cuda.is_available')
def test_emotion_embedding_init_cuda(
    mock_cuda, mock_feature_extractor, mock_model
):
    """Test EmotionEmbedding initialization with CUDA available."""
    mock_cuda.return_value = True
    mock_feature_extractor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance

    # Initialize EmotionEmbedding
    emotion_embedding = EmotionEmbedding(ser_model_name=DUMMY_SER_MODEL_NAME)

    # Verify models were loaded
    mock_feature_extractor.assert_called_once_with(DUMMY_SER_MODEL_NAME)
    mock_model.assert_called_once_with(DUMMY_SER_MODEL_NAME)

    # Verify model was moved to GPU
    mock_model_instance.to.assert_called_once()

    # Verify device is set correctly
    assert emotion_embedding.device.type == 'cuda'


@patch(
    'vecl.embeddings.emotion.AutoModelForAudioClassification.from_pretrained'
)
@patch('vecl.embeddings.emotion.AutoFeatureExtractor.from_pretrained')
@patch('vecl.embeddings.emotion.torch.cuda.is_available')
def test_emotion_embedding_init_cpu(
    mock_cuda, mock_feature_extractor, mock_model
):
    """Test EmotionEmbedding initialization without CUDA."""
    mock_cuda.return_value = False
    mock_feature_extractor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance

    # Initialize EmotionEmbedding
    emotion_embedding = EmotionEmbedding(ser_model_name=DUMMY_SER_MODEL_NAME)

    # Verify device is set to CPU
    assert emotion_embedding.device.type == 'cpu'


@patch('vecl.embeddings.emotion.torchaudio.load')
@patch(
    'vecl.embeddings.emotion.AutoModelForAudioClassification.from_pretrained'
)
@patch('vecl.embeddings.emotion.AutoFeatureExtractor.from_pretrained')
def test_get_emotion_embedding_basic(
    mock_feature_extractor,
    mock_model,
    mock_audio_load,
    mock_audio_tensor,
    mock_model_outputs,
):
    """Test basic emotion embedding extraction."""
    # Setup mocks
    mock_feature_extractor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance

    # Mock audio loading
    mock_audio_load.return_value = (mock_audio_tensor, 16000)

    # Mock feature extractor
    mock_feature_extractor_instance.sampling_rate = 16000
    mock_inputs = {'input_values': torch.randn(1, 16000)}
    mock_feature_extractor_instance.return_value = mock_inputs

    # Mock model outputs
    mock_model_instance.return_value = mock_model_outputs

    # Initialize and test
    emotion_embedding = EmotionEmbedding(ser_model_name=DUMMY_SER_MODEL_NAME)
    result = emotion_embedding.get_emotion_embedding('/fake/path/audio.wav')

    # Verify result shape and type
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1024)  # Mean pooled embedding


@patch('vecl.embeddings.emotion.torchaudio.load')
@patch(
    'vecl.embeddings.emotion.AutoModelForAudioClassification.from_pretrained'
)
@patch('vecl.embeddings.emotion.AutoFeatureExtractor.from_pretrained')
def test_get_emotion_embedding_resampling(
    mock_feature_extractor, mock_model, mock_audio_load, mock_model_outputs
):
    """Test emotion embedding with audio resampling."""
    # Setup mocks
    mock_feature_extractor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance

    # Mock audio loading with different sample rate
    audio_tensor = torch.randn(1, 22050)  # 22kHz audio
    mock_audio_load.return_value = (audio_tensor, 22050)

    # Mock feature extractor expecting 16kHz
    mock_feature_extractor_instance.sampling_rate = 16000
    mock_inputs = {'input_values': torch.randn(1, 16000)}
    mock_feature_extractor_instance.return_value = mock_inputs

    # Mock model outputs
    mock_model_instance.return_value = mock_model_outputs

    # Test - should handle resampling internally
    emotion_embedding = EmotionEmbedding(ser_model_name=DUMMY_SER_MODEL_NAME)

    with patch(
        'vecl.embeddings.emotion.torchaudio.transforms.Resample'
    ) as mock_resample:
        mock_resampler = MagicMock()
        mock_resample.return_value = mock_resampler
        mock_resampler.return_value = torch.randn(1, 16000)  # Resampled audio

        result = emotion_embedding.get_emotion_embedding(
            '/fake/path/audio.wav'
        )

        # Verify resampling was used
        mock_resample.assert_called_once_with(orig_freq=22050, new_freq=16000)
        assert isinstance(result, torch.Tensor)


@patch('vecl.embeddings.emotion.torchaudio.load')
@patch(
    'vecl.embeddings.emotion.AutoModelForAudioClassification.from_pretrained'
)
@patch('vecl.embeddings.emotion.AutoFeatureExtractor.from_pretrained')
def test_get_emotion_embedding_stereo_to_mono(
    mock_feature_extractor, mock_model, mock_audio_load, mock_model_outputs
):
    """Test emotion embedding with stereo audio conversion to mono."""
    # Setup mocks
    mock_feature_extractor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_feature_extractor.return_value = mock_feature_extractor_instance
    mock_model.return_value = mock_model_instance

    # Mock stereo audio loading
    stereo_audio = torch.randn(2, 16000)  # 2 channels
    mock_audio_load.return_value = (stereo_audio, 16000)

    mock_feature_extractor_instance.sampling_rate = 16000
    mock_inputs = {'input_values': torch.randn(1, 16000)}
    mock_feature_extractor_instance.return_value = mock_inputs

    # Mock model outputs
    mock_model_instance.return_value = mock_model_outputs

    # Test - should convert stereo to mono
    emotion_embedding = EmotionEmbedding(ser_model_name=DUMMY_SER_MODEL_NAME)
    result = emotion_embedding.get_emotion_embedding('/fake/path/audio.wav')

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1024)


def test_compute_emotion_embeddings_existing_file(
    temp_dataset_dir, mock_dataset_configs
):
    """Test when emotion embeddings file already exists."""
    embeddings_file = temp_dataset_dir / 'emotion_embeddings.pth'
    embeddings_file.touch()  # Create empty file to simulate existing embeddings

    # Call function
    result = compute_emotion_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        ser_model_name=DUMMY_SER_MODEL_NAME,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    # Should return early without processing
    assert result is None
    assert embeddings_file.exists()


@patch('vecl.embeddings.emotion.download_s3_file')
@patch('vecl.embeddings.emotion.torch.save')
@patch('vecl.embeddings.emotion.load_tts_samples')
@patch('vecl.embeddings.emotion.EmotionEmbedding')
def test_compute_emotion_embeddings_success(
    mock_emotion_embedding_class,
    mock_load_samples,
    mock_torch_save,
    mock_download,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test successful emotion embeddings computation."""
    embeddings_file = temp_dataset_dir / 'emotion_embeddings.pth'

    # Setup mocks
    mock_load_samples.return_value = (sample_tts_samples, None)

    mock_emotion_embedder = MagicMock()
    mock_emotion_embedding_class.return_value = mock_emotion_embedder

    # Mock embedding computation
    def mock_get_embedding(audio_file):
        return torch.randn(1, 1024)  # Fake embedding

    mock_emotion_embedder.get_emotion_embedding.side_effect = (
        mock_get_embedding
    )

    # Call function
    compute_emotion_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        ser_model_name=DUMMY_SER_MODEL_NAME,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    # Verify download was attempted
    mock_download.assert_called_once()

    # Verify TTS samples were loaded
    mock_load_samples.assert_called_once_with(
        mock_dataset_configs, eval_split=False
    )

    # Verify EmotionEmbedding was instantiated
    mock_emotion_embedding_class.assert_called_once_with(
        ser_model_name=DUMMY_SER_MODEL_NAME
    )

    # Verify embeddings were computed for each audio file
    assert mock_emotion_embedder.get_emotion_embedding.call_count == 5

    # Verify torch.save was called
    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args[0]
    emotion_embeddings = call_args[0]
    saved_path = call_args[1]

    # Verify the emotion_embeddings mapping
    assert len(emotion_embeddings) == 5
    for sample in sample_tts_samples:
        relative_path = os.path.relpath(
            sample['audio_file'], sample['root_path']
        )
        assert relative_path in emotion_embeddings
    assert saved_path == embeddings_file


@patch('vecl.embeddings.emotion.download_s3_file')
@patch('vecl.embeddings.emotion.torch.save')
@patch('vecl.embeddings.emotion.load_tts_samples')
@patch('vecl.embeddings.emotion.EmotionEmbedding')
def test_compute_emotion_embeddings_with_errors(
    mock_emotion_embedding_class,
    mock_load_samples,
    mock_torch_save,
    mock_download,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test emotion embeddings computation with some failures."""
    embeddings_file = temp_dataset_dir / 'emotion_embeddings.pth'

    mock_load_samples.return_value = (sample_tts_samples, None)

    mock_emotion_embedder = MagicMock()
    mock_emotion_embedding_class.return_value = mock_emotion_embedder

    # Mock embedding computation with some failures
    def mock_get_embedding_with_error(audio_file):
        if 'coraaa-486-00000' in audio_file:
            raise Exception('Failed to process audio_001')
        return torch.randn(1, 1024)

    mock_emotion_embedder.get_emotion_embedding.side_effect = (
        mock_get_embedding_with_error
    )

    # Call function (should not raise exception)
    compute_emotion_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        ser_model_name=DUMMY_SER_MODEL_NAME,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    # Verify torch.save was still called
    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args[0]
    emotion_embeddings = call_args[0]

    # Should only have embeddings for audio_002 and audio_003 (audio_001 failed)
    assert len(emotion_embeddings) == 4
    assert 'audio/coraaa-904-00001.wav' in emotion_embeddings
    assert 'audio/coraaa-874-00002.wav' in emotion_embeddings
    assert 'audio/coraaa-486-00000.wav' not in emotion_embeddings


def test_compute_emotion_embeddings_empty_samples(
    temp_dataset_dir, mock_dataset_configs
):
    """Test emotion embeddings computation with no samples."""
    embeddings_file = temp_dataset_dir / 'emotion_embeddings.pth'

    with (
        patch('vecl.embeddings.emotion.download_s3_file'),
        patch('vecl.embeddings.emotion.load_tts_samples') as mock_load_samples,
        patch(
            'vecl.embeddings.emotion.EmotionEmbedding'
        ) as mock_emotion_embedding_class,
        patch('vecl.embeddings.emotion.torch.save') as mock_torch_save,
    ):
        mock_load_samples.return_value = ([], None)  # No samples
        mock_emotion_embedding_class.return_value = MagicMock()

        # Call function
        compute_emotion_embeddings(
            dataset_configs=mock_dataset_configs,
            embeddings_file_path=embeddings_file,
            ser_model_name=DUMMY_SER_MODEL_NAME,
            s3_bucket='dummy-bucket',
            s3_key='dummy-key',
        )

        # Should still save (empty) embeddings
        mock_torch_save.assert_called_once()
        call_args = mock_torch_save.call_args[0]
        emotion_embeddings = call_args[0]
        assert len(emotion_embeddings) == 0


@patch('vecl.embeddings.emotion.download_s3_file')
@patch('vecl.embeddings.emotion.os.path.relpath', side_effect=os.path.relpath)
@patch('vecl.embeddings.emotion.torch.save')
@patch('vecl.embeddings.emotion.load_tts_samples')
@patch('vecl.embeddings.emotion.EmotionEmbedding')
def test_relative_path_computation(
    mock_emotion_embedding_class,
    mock_load_samples,
    mock_torch_save,
    mock_relpath,
    mock_download,
    temp_dataset_dir,
    mock_dataset_configs,
    sample_tts_samples,
):
    """Test that relative paths are computed correctly."""
    embeddings_file = temp_dataset_dir / 'emotion_embeddings.pth'

    mock_load_samples.return_value = (sample_tts_samples, None)
    mock_emotion_embedder = MagicMock()
    mock_emotion_embedding_class.return_value = mock_emotion_embedder
    mock_emotion_embedder.get_emotion_embedding.return_value = torch.randn(
        1, 1024
    )

    # Call function
    compute_emotion_embeddings(
        dataset_configs=mock_dataset_configs,
        embeddings_file_path=embeddings_file,
        ser_model_name=DUMMY_SER_MODEL_NAME,
        s3_bucket='dummy-bucket',
        s3_key='dummy-key',
    )

    # Verify relative path was computed for each sample
    assert mock_relpath.call_count == 5
    expected_calls = [
        (
            'data/processed_24k/audio/coraaa-486-00000.wav',
            'data/processed_24k',
        ),
        (
            'data/processed_24k/audio/coraaa-904-00001.wav',
            'data/processed_24k',
        ),
        (
            'data/processed_24k/audio/coraaa-874-00002.wav',
            'data/processed_24k',
        ),
        (
            'data/processed_24k/audio/coraaa-717-00003.wav',
            'data/processed_24k',
        ),
        (
            'data/processed_24k/audio/coraaa-529-00999.wav',
            'data/processed_24k',
        ),
    ]

    actual_calls = mock_relpath.call_args_list
    assert [c.args for c in actual_calls] == expected_calls
