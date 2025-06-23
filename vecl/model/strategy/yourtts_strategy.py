from pathlib import Path
from typing import Optional, Tuple

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits

from vecl.model.strategy.base import BaseModelStrategy


class YourTTSStrategy(BaseModelStrategy):
    """Strategy for handling YourTTS models."""

    def get_checkpoint_prefix_s3(self) -> Optional[str]:
        if self.config.s3:
            return self.config.s3.checkpoint_prefix_yourtts
        return None

    def patch_config_for_training(
        self, model_config: VitsConfig, dataset_configs: list
    ) -> VitsConfig:
        # For YourTTS, we mainly need to ensure the d_vector (speaker embedding)
        # settings are correctly propagated from the main AppConfig.
        if self.config.training.use_d_vector_file:
            d_vector_files = (
                [str(p) for p in self.config.training.d_vector_file]
                if self.config.training.d_vector_file
                else None
            )
            model_config.model_args.d_vector_file = d_vector_files
            model_config.d_vector_file = d_vector_files  # For SpeakerManager
            model_config.model_args.use_d_vector_file = True
        else:
            model_config.model_args.d_vector_file = None
            model_config.d_vector_file = None
            model_config.model_args.use_d_vector_file = False

        # Ensure VECL-specific features are disabled.
        if hasattr(model_config.model_args, 'use_emotion_consistency_loss'):
            model_config.model_args.use_emotion_consistency_loss = False
        if hasattr(model_config.model_args, 'emotion_embedding_file'):
            model_config.model_args.emotion_embedding_file = None

        return model_config

    def init_model_from_config(
        self, model_config: VitsConfig
    ) -> Tuple[Vits, VitsConfig]:
        model = Vits.init_from_config(model_config)
        return model, model_config

    def get_checkpoint_path_for_inference(self) -> Path:
        return (
            self.config.paths.pretrained_checkpoint_dir
            / self.config.s3.cml_tts_checkpoint_key
        )
