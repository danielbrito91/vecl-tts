from pathlib import Path
from typing import Optional, Tuple

from TTS.tts.configs.vits_config import VitsConfig

from vecl.model.config import VeclArgs, VeclConfig
from vecl.model.strategy.base import BaseModelStrategy
from vecl.model.vecl import Vecl


class VeclStrategy(BaseModelStrategy):
    """Strategy for handling VECL models."""

    def get_checkpoint_prefix_s3(self) -> Optional[str]:
        if self.config.s3:
            return self.config.s3.checkpoint_prefix_vecl
        return None

    def patch_config_for_training(
        self, model_config: VitsConfig, dataset_configs: list
    ) -> VeclConfig:
        # Monkey-patch the classes to our custom VECL versions.
        model_config.__class__ = VeclConfig
        if hasattr(model_config, 'model_args'):
            model_config.model_args.__class__ = VeclArgs
        else:
            model_config.model_args = VeclArgs()

        # Apply VECL-specific model_args from the main AppConfig.
        model_config.model_args.use_emotion_consistency_loss = (
            self.config.model.use_emotion_consistency_loss
        )
        model_config.model_args.emotion_embedding_file = str(
            self.config.paths.emotion_embeddings_file
        )
        model_config.model_args.use_d_vector_file = (
            self.config.training.use_d_vector_file
        )

        if (
            self.config.training.use_d_vector_file
            and not self.config.training.d_vector_file
        ):
            print(
                '⚠️ Warning: use_d_vector_file is True but d_vector_file is None'
            )

        model_config.model_args.d_vector_file = (
            [str(p) for p in self.config.training.d_vector_file]
            if self.config.training.d_vector_file
            else None
        )

        model_config.d_vector_file = model_config.model_args.d_vector_file
        return model_config

    def init_model_from_config(
        self, model_config: VeclConfig
    ) -> Tuple[Vecl, VeclConfig]:
        return Vecl.init_from_config(model_config)

    def get_checkpoint_path_for_inference(self) -> Path:
        return self.config.paths.restore_path
