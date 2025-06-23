"""
Defines the strategy pattern for handling different model types like VECL and
YourTTS.
This allows for easy extension to new model types without modifying core
training and inference logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

from vecl.config import AppConfig


class BaseModelStrategy(ABC):
    """
    Abstract base class for a model strategy. It defines the interface for
    model-specific operations.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def get_checkpoint_prefix_s3(self) -> Optional[str]:
        """
        Returns the S3 prefix for checkpoints based on the model type. Default
        implementation returns None.
        """
        return None

    @abstractmethod
    def patch_config_for_training(
        self, model_config: VitsConfig, dataset_configs: list
    ) -> VitsConfig:
        """
        Patches the loaded model configuration with training-specific settings.
        """

    @abstractmethod
    def init_model_from_config(
        self, model_config: VitsConfig
    ) -> Tuple[Vits, VitsConfig]:
        """
        Initializes a model instance from a configuration object.
        """

    @abstractmethod
    def get_checkpoint_path_for_inference(self) -> Path:
        """
        Returns the path to the model checkpoint for inference.
        """
