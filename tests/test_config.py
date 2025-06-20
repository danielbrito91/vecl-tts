from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig
from omegaconf.errors import InterpolationResolutionError

from vecl.config import AppConfig


def test_hydra_config_composition(monkeypatch):
    """
    Tests that Hydra correctly composes the default config and that it can be
    validated by Pydantic.
    """
    # Set required environment variables for the test
    monkeypatch.setenv('OUTPUT_PATH', '/tmp/output')
    monkeypatch.setenv('DATASET_PATH', '/tmp/data')
    monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')

    # Initialize Hydra and compose the configuration
    with hydra.initialize(config_path='../configs', version_base=None):
        cfg: DictConfig = hydra.compose(config_name='config')

        # Validate with Pydantic
        app_config = AppConfig(**cfg)

    # Assertions
    assert app_config.model.type == 'vecl'
    assert app_config.paths.output_path == Path('/tmp/output')
    assert app_config.training.batch_size == 12
    assert app_config.s3.bucket_name == 'test-bucket'


def test_hydra_model_override(monkeypatch):
    """
    Tests that Hydra model overrides work as expected.
    """
    monkeypatch.setenv('OUTPUT_PATH', '/tmp/output')
    monkeypatch.setenv('DATASET_PATH', '/tmp/data')
    monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')

    with hydra.initialize(config_path='../configs', version_base=None):
        # Compose with a model override
        cfg: DictConfig = hydra.compose(
            config_name='config', overrides=['model=yourtts']
        )
        app_config = AppConfig(**cfg)

    # Assert that the model type was correctly overridden
    assert app_config.model.type == 'yourtts'
    assert not app_config.model.use_emotion_consistency_loss


def test_hydra_fails_on_missing_env_var(monkeypatch):
    """
    Tests that Hydra raises an InterpolationResolutionError if a required
    environment variable is not set.
    """
    # Ensure a required variable is NOT set
    monkeypatch.delenv('OUTPUT_PATH', raising=False)
    monkeypatch.setenv('DATASET_PATH', '/tmp/data')
    monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')

    with hydra.initialize(config_path='../configs', version_base=None):
        cfg = hydra.compose(config_name='config')
        with pytest.raises(InterpolationResolutionError):
            # Accessing the value is required to trigger Hydra's lazy interpolation
            _ = cfg.paths.output_path
