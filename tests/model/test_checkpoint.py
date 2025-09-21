"""Tests for model checkpoint loading and path resolution."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from vecl.models.strategy.factory import get_model_strategy

from vecl.models.loader import (
    _resolve_speaker_encoder_paths,
    load_model_for_inference,
    load_model_for_training,
)
from vecl.models.vecl import Vecl


@pytest.fixture
def checkpoint_test_config(hydra_test_config: DictConfig) -> DictConfig:
    """Fixture to create a config and directory structure for checkpoint tests."""
    config = hydra_test_config

    # Create directories from the config paths for a clean test environment
    output_path = Path(config.paths.output_path)
    se_model_dir = Path(config.paths.speaker_encoder_model_dir)
    pretrained_dir = Path(config.paths.pretrained_checkpoint_dir)
    restore_dir = Path(config.paths.restore_path).parent

    for p in [output_path, se_model_dir, pretrained_dir, restore_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return config


# Tests for the private helper `_resolve_speaker_encoder_paths`
def test_resolve_speaker_encoder_paths_already_exists(checkpoint_test_config):
    """
    Given: Speaker encoder model and config files already exist in the target directory.
    When: _resolve_speaker_encoder_paths is called.
    Then: It should return the correct paths to the existing files without any changes.
    """
    config = checkpoint_test_config
    se_dir = Path(config.paths.speaker_encoder_model_dir)

    # Pre-create the files
    (se_dir / 'model_se.pth').touch()
    (se_dir / 'config_se.json').touch()

    model_path, config_path = _resolve_speaker_encoder_paths(config)

    assert model_path == se_dir / 'model_se.pth'
    assert config_path == se_dir / 'config_se.json'
    assert model_path.is_file()
    assert config_path.is_file()


def test_resolve_speaker_encoder_paths_moves_from_restore_dir(
    checkpoint_test_config,
):
    """
    Given: Speaker encoder files exist in the checkpoint restore directory but not the target.
    When: _resolve_speaker_encoder_paths is called.
    Then: It should move the files to the target directory and return the new paths.
    """
    config = checkpoint_test_config
    se_dir = Path(config.paths.speaker_encoder_model_dir)
    restore_dir = Path(config.paths.restore_path).parent

    # Create files in the source (restore) directory
    (restore_dir / 'model_se.pth').touch()
    (restore_dir / 'config_se.json').touch()

    # Ensure files do not exist in the destination
    assert not (se_dir / 'model_se.pth').exists()
    assert not (se_dir / 'config_se.json').exists()

    model_path, config_path = _resolve_speaker_encoder_paths(config)

    # Check that paths point to the target directory
    assert model_path == se_dir / 'model_se.pth'
    assert config_path == se_dir / 'config_se.json'

    # Check that files now exist in the target and are gone from the source
    assert model_path.is_file()
    assert config_path.is_file()
    assert not (restore_dir / 'model_se.pth').exists()
    assert not (restore_dir / 'config_se.json').exists()


@patch('vecl.model.checkpoint.download_s3_file')
def test_resolve_speaker_encoder_paths_downloads_from_s3(
    mock_download, checkpoint_test_config
):
    """
    Given: Speaker encoder files are not available locally.
    When: _resolve_speaker_encoder_paths is called.
    Then: It should attempt to download them from S3.
    """
    config = checkpoint_test_config
    se_dir = Path(config.paths.speaker_encoder_model_dir)

    def downloader_side_effect(bucket_name, s3_key, local_path):
        """Simulate the download by creating a dummy file."""
        local_path.touch()

    mock_download.side_effect = downloader_side_effect

    model_path, config_path = _resolve_speaker_encoder_paths(config)

    assert model_path == se_dir / 'model_se.pth'
    assert config_path == se_dir / 'config_se.json'
    assert model_path.is_file()
    assert config_path.is_file()
    assert mock_download.call_count == 2
    mock_download.assert_any_call(
        bucket_name=config.s3.bucket_name,
        s3_key='speaker_encoder/model_se.pth',
        local_path=model_path,
    )
    mock_download.assert_any_call(
        bucket_name=config.s3.bucket_name,
        s3_key='speaker_encoder/config_se.json',
        local_path=config_path,
    )


@patch(
    'vecl.model.checkpoint.download_s3_file',
    side_effect=Exception('S3 download failed'),
)
def test_resolve_speaker_encoder_paths_file_not_found_raises(
    mock_download, checkpoint_test_config
):
    """
    Given: Speaker encoder files are not found locally and S3 download fails.
    When: _resolve_speaker_encoder_paths is called.
    Then: It should raise a FileNotFoundError.
    """
    config = checkpoint_test_config
    with pytest.raises(FileNotFoundError):
        _resolve_speaker_encoder_paths(config)


@pytest.fixture
def model_setup(checkpoint_test_config: DictConfig):
    """
    Fixture to create the basic file structure for model loading tests,
    without the main checkpoint file itself. This can be used for both
    training and inference test setups.
    """
    config = checkpoint_test_config
    restore_dir = Path(config.paths.restore_path).parent

    # Create a dummy model config file, as this is required for all model loads
    dummy_model_config = {
        'model': 'YourTTS',
        'model_args': {
            'num_lang': 1,
            'd_vector_file': 'speakers.pth',
            'speaker_encoder_model_path': 'se/model.pth',
            'speaker_encoder_config_path': 'se/config.json',
            'emotion_embedding_dim': 128,  # Mark as VECL for inference test
        },
        'audio': {},
    }
    config_path = restore_dir / 'config.json'
    config_path.write_text(json.dumps(dummy_model_config))
    config.paths.pretrained_config_path = str(config_path)

    # Create dummy language IDs file
    (restore_dir / 'language_ids.json').write_text(json.dumps({'en': 0}))

    # Create dummy speaker encoder files to satisfy path resolution
    se_dir = Path(config.paths.speaker_encoder_model_dir)
    (se_dir / 'model_se.pth').touch()
    (se_dir / 'config_se.json').touch()

    return config


@pytest.fixture
def training_setup(model_setup: DictConfig):
    """Prepares a valid environment for `load_model_for_training` tests."""
    config = model_setup
    restore_path = Path(config.paths.restore_path)

    # Create a dummy checkpoint file
    dummy_checkpoint = {
        'model': {
            'layer.weight': torch.randn(1, 1),
            'emb_l.weight': torch.randn(2, 2),
        }
    }
    torch.save(dummy_checkpoint, restore_path)

    # Create dummy dataset configs
    dataset_configs = [
        OmegaConf.create({
            'name': 'dummy_dataset',
            'path': '/dummy/path',
            'meta_file_train': 'meta.csv',
        })
    ]
    return config, dataset_configs


def mock_model_strategy():
    """Helper to create a mock model strategy for patching."""
    strategy = MagicMock()
    model = MagicMock(spec=Vecl)
    model.load_state_dict = MagicMock()
    model_config = OmegaConf.create({
        'model': 'test_model',
        'model_args': {
            'speaker_encoder_model_path': '',
            'speaker_encoder_config_path': '',
        },
    })

    strategy.patch_config_for_training.return_value = model_config
    strategy.init_model_from_config.return_value = (model, model_config)
    return strategy, model, model_config


# Tests for `load_model_for_training`
@patch('vecl.model.checkpoint.get_model_strategy')
@patch('vecl.model.checkpoint.patch_state_dict', side_effect=lambda x: x)
def test_load_model_for_training_happy_path(
    mock_patch_dict, mock_get_strategy, training_setup
):
    """
    Given: A valid training setup with all required files present.
    When: load_model_for_training is called.
    Then: It should load the model successfully, patching configs and state dicts correctly.
    """
    config, dataset_configs = training_setup
    mock_strategy, mock_model, mock_model_config = mock_model_strategy()
    mock_get_strategy.return_value = mock_strategy

    model = load_model_for_training(config, dataset_configs)

    assert model is mock_model
    mock_get_strategy.assert_called_once_with(config)
    mock_strategy.patch_config_for_training.assert_called_once()
    mock_strategy.init_model_from_config.assert_called_once()
    mock_model.load_state_dict.assert_called_once()
    assert mock_model_config.batch_size == config.training.batch_size
    assert mock_model_config.output_path == str(config.paths.output_path)


@patch('vecl.model.checkpoint.get_model_strategy')
@patch('vecl.model.checkpoint.patch_state_dict', side_effect=lambda x: x)
def test_load_model_for_training_no_pretrained_lang_embs(
    mock_patch_dict, mock_get_strategy, training_setup
):
    """
    Given: The config specifies not to use pretrained language embeddings.
    When: load_model_for_training is called.
    Then: It should remove the 'emb_l.weight' key from the state dict before loading.
    """
    config, dataset_configs = training_setup
    config.training.use_pretrained_lang_embeddings = False

    mock_strategy, mock_model, _ = mock_model_strategy()
    mock_get_strategy.return_value = mock_strategy

    load_model_for_training(config, dataset_configs)

    loaded_state_dict = mock_model.load_state_dict.call_args[0][0]
    assert 'emb_l.weight' not in loaded_state_dict


@patch('vecl.model.checkpoint.get_model_strategy')
@patch('vecl.utils.downloader.extract_tar_file')
def test_load_model_for_training_extracts_local_tar(
    mock_extract, mock_get_strategy, model_setup
):
    """
    Given: The checkpoint is missing, but a local tarball exists.
    When: load_model_for_training is called.
    Then: It should extract the tarball to find the checkpoint.
    """
    config = model_setup
    restore_path = Path(config.paths.restore_path)
    Path(config.paths.local_tar_path).touch()

    def extract_side_effect(tar_path, dest_dir):
        """Simulate extraction by creating the checkpoint file."""
        torch.save({'model': {}}, restore_path)

    mock_extract.side_effect = extract_side_effect
    mock_get_strategy.return_value = mock_model_strategy()[0]

    with patch(
        'vecl.model.checkpoint.patch_state_dict', side_effect=lambda x: x
    ):
        load_model_for_training(config, [])

    mock_extract.assert_called_once_with(
        Path(config.paths.local_tar_path),
        Path(config.paths.pretrained_checkpoint_dir),
    )


def test_load_model_for_training_missing_checkpoint_raises(model_setup):
    """
    Given: No checkpoint, no tarball, and no S3 mock.
    When: load_model_for_training is called.
    Then: It should raise FileNotFoundError.
    """
    config = model_setup
    # Patch S3 download to prevent it from running and failing the test setup
    with (
        patch(
            'vecl.model.checkpoint.download_s3_file',
            side_effect=FileNotFoundError,
        ),
        pytest.raises(FileNotFoundError),
    ):
        load_model_for_training(config, [])


# Tests for `load_model_for_inference`
@pytest.fixture
def inference_setup(model_setup: DictConfig):
    """Prepares a valid environment for `load_model_for_inference` tests."""
    config = model_setup
    config.model.type = 'vecl'  # Ensure strategy selects a checkpoint path

    strategy = get_model_strategy(config)
    checkpoint_path = strategy.get_checkpoint_path_for_inference()
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            'model': {
                'layer.weight': torch.randn(1, 1),
                'emotion_proj.proj.weight': torch.randn(128, 512),
            }
        },
        checkpoint_path,
    )
    return config, checkpoint_path


@patch('vecl.model.checkpoint.Vecl.init_from_config')
@patch('vecl.model.checkpoint.setup_model')
def test_load_model_for_inference_happy_path(
    mock_setup_model, mock_init_vecl, inference_setup
):
    """
    Given: A valid inference setup with a VECL model checkpoint.
    When: load_model_for_inference is called.
    Then: It should load the VECL model, patch the state dict, and prepare it for inference.
    """
    config, _ = inference_setup
    device = torch.device('cpu')

    mock_model = MagicMock(spec=Vecl)
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_init_vecl.return_value = (mock_model,)

    model, cfg = load_model_for_inference(config, device)

    assert model is mock_model
    mock_init_vecl.assert_called_once()
    mock_setup_model.assert_not_called()  # Ensure non-VECL path is not taken
    mock_model.load_state_dict.assert_called_once()
    mock_model.eval.assert_called_once()
    mock_model.to.assert_called_with(device)
    assert cfg.model_args.d_vector_file is None  # Check config massaging


def test_load_model_for_inference_missing_checkpoint_raises(model_setup):
    """
    Given: The model checkpoint for inference is missing.
    When: load_model_for_inference is called.
    Then: It should raise a FileNotFoundError.
    """
    config = model_setup
    config.model.type = 'vecl'
    device = torch.device('cpu')

    with pytest.raises(FileNotFoundError):
        load_model_for_inference(config, device)
