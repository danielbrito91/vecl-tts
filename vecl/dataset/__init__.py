"""Dataset utilities for VECL-TTS."""

from .preparation import prepare_dataset_configs
from .vecl_dataset import VeclDataset

__all__ = ['prepare_dataset_configs', 'VeclDataset']
