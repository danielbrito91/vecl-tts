import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from TTS.config import load_config as coqui_load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager

from vecl.config import AppConfig
from vecl.models.layers import EmotionProj
from vecl.models.vecl import Vecl
from vecl.training.utils import patch_state_dict

logger = logging.getLogger(__name__)


class ModelLoader(ABC):
    """Base class for model loading."""

    def __init__(self, config: AppConfig):
        self.config = config

    def load_for_training(self, dataset_configs: list):
        """Load model for training."""
        model_config = self._create_training_config()
        model_config.datasets = dataset_configs

        model = self._init_model(model_config)
        self._load_weights(model)
        return model

    def load_for_inference(self, device: torch.device):
        """Load model for inference."""
        checkpoint_path = self._get_inference_checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        cfg = self._create_inference_config(checkpoint_path)
        model = self._init_model_for_inference(cfg)
        self._load_inference_weights(model, checkpoint_path)

        return model.eval().to(device), cfg

    def _create_training_config(self):
        """Create config for training."""
        # Find config file
        config_path = self.config.paths.pretrained_config_path
        if not config_path.exists():
            config_path = self.config.paths.restore_path.parent / 'config.json'

        model_config = coqui_load_config(config_path)

        # Apply common training settings
        model_config.output_path = str(self.config.paths.output_path)
        model_config.run_name = self.config.training.run_name
        model_config.batch_size = self.config.training.batch_size
        model_config.eval_batch_size = self.config.training.eval_batch_size
        model_config.num_loader_workers = (
            self.config.training.num_loader_workers
        )
        model_config.epochs = self.config.training.epochs
        model_config.lr = self.config.training.learning_rate
        model_config.save_step = self.config.training.save_step
        model_config.max_text_len = self.config.training.max_text_len

        # Configure dashboard logger (wandb or tensorboard)
        if self.config.wandb:
            # Enable W&B in Coqui trainer config
            model_config.dashboard_logger = 'wandb'
            model_config.project_name = self.config.wandb.project_name
            # run_name already set above
            model_config.logger_uri = self.config.wandb.entity
        else:
            model_config.dashboard_logger = 'tensorboard'

        # Audio settings
        model_config.audio.sample_rate = self.config.audio.sample_rate
        model_config.audio.max_audio_len = int(
            self.config.audio.max_audio_len_seconds
            * self.config.audio.sample_rate
        )

        # Language file
        language_ids_path = (
            self.config.paths.restore_path.parent / 'language_ids.json'
        )
        if language_ids_path.exists():
            model_config.language_ids_file = str(language_ids_path)
            if hasattr(model_config, 'model_args'):
                model_config.model_args.language_ids_file = str(
                    language_ids_path
                )

        # Make speaker encoder paths absolute
        checkpoint_dir = self.config.paths.restore_path.parent
        if (
            hasattr(model_config, 'model_args')
            and hasattr(model_config.model_args, 'speaker_encoder_config_path')
            and model_config.model_args.speaker_encoder_config_path
        ):
            se_config_path = Path(
                model_config.model_args.speaker_encoder_config_path
            )
            model_config.model_args.speaker_encoder_config_path = str(
                checkpoint_dir / se_config_path.name
            )

        if (
            hasattr(model_config, 'model_args')
            and hasattr(model_config.model_args, 'speaker_encoder_model_path')
            and model_config.model_args.speaker_encoder_model_path
        ):
            se_model_path = Path(
                model_config.model_args.speaker_encoder_model_path
            )
            model_config.model_args.speaker_encoder_model_path = str(
                checkpoint_dir / se_model_path.name
            )

        # Speaker embeddings
        if self.config.training.use_d_vector_file:
            model_config.model_args.d_vector_file = [
                str(self.config.paths.speaker_embeddings_file)
            ]
            model_config.d_vector_file = [
                str(self.config.paths.speaker_embeddings_file)
            ]
            model_config.model_args.use_d_vector_file = True

        # Apply model-specific settings
        return self._patch_config_for_training(model_config)

    def _create_inference_config(self, checkpoint_path: Path):
        """Create config for inference."""
        config_path = checkpoint_path.parent / 'config.json'
        cfg = coqui_load_config(config_path)

        # Basic setup
        cfg.audio.sample_rate = 24_000
        cfg.model_args.d_vector_file = None
        cfg.model_args.use_speaker_encoder_as_loss = False

        # Language setup
        language_ids_path = checkpoint_path.parent / 'language_ids.json'
        if language_ids_path.exists():
            cfg.language_ids_file = str(language_ids_path)
            cfg.model_args.language_ids_file = str(language_ids_path)

        return cfg

    def _load_weights(self, model):
        """Load training weights."""
        # Load to CPU first for compatibility and memory efficiency
        checkpoint = torch.load(
            self.config.paths.restore_path, map_location='cpu'
        )
        state_dict = patch_state_dict(checkpoint['model'])

        # Remove language embeddings if not using pretrained ones
        if (
            not self.config.training.use_pretrained_lang_embeddings
            and 'emb_l.weight' in state_dict
        ):
            del state_dict['emb_l.weight']

        model.load_state_dict(state_dict, strict=False)

    def _load_inference_weights(self, model, checkpoint_path: Path):
        """Load inference weights."""
        # Load to CPU first for compatibility, then model.to(device) handles final placement
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']

        # Clean up
        state_dict = {
            k: v for k, v in state_dict.items() if 'speaker_encoder' not in k
        }

        # Handle language embedding size mismatch
        if 'emb_l.weight' in state_dict and hasattr(model, 'emb_l'):
            if state_dict['emb_l.weight'].shape != model.emb_l.weight.shape:
                del state_dict['emb_l.weight']

        # Create emotion projection if needed
        if 'emotion_proj.proj.weight' in state_dict and not hasattr(
            model, 'emotion_proj'
        ):
            w_shape = state_dict['emotion_proj.proj.weight'].shape
            model.emotion_proj = EmotionProj(w_shape[1], w_shape[0])

        # Handle new fusion layer missing in older checkpoints
        if 'emotion_fusion.weight' not in state_dict and hasattr(
            model, 'emotion_fusion'
        ):
            # Nothing to load; keep randomly initialized fusion layer
            pass

        state_dict = patch_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)

        # Language manager
        language_ids_path = checkpoint_path.parent / 'language_ids.json'
        if language_ids_path.exists():
            model.language_manager = LanguageManager(language_ids_path)

    @abstractmethod
    def _patch_config_for_training(self, model_config):
        """Apply model-specific config patches for training."""
        pass

    @abstractmethod
    def _init_model(self, model_config):
        """Initialize model from config."""
        pass

    @abstractmethod
    def _init_model_for_inference(self, cfg):
        """Initialize model for inference."""
        pass

    @abstractmethod
    def _get_inference_checkpoint_path(self) -> Path:
        """Get the checkpoint path for inference."""
        pass


class VeclLoader(ModelLoader):
    """Loader for VECL models."""

    def _patch_config_for_training(self, model_config):
        # Simply add the needed fields - no class monkey-patching
        model_config.use_emotion_consistency_loss = (
            self.config.model.use_emotion_consistency_loss
        )
        model_config.emotion_embedding_file = str(
            self.config.paths.emotion_embeddings_file
        )

        # Ensure model_args has the VECL fields
        if not hasattr(model_config.model_args, 'emotion_embedding_dim'):
            model_config.model_args.emotion_embedding_dim = 768
        if not hasattr(
            model_config.model_args, 'use_emotion_consistency_loss'
        ):
            model_config.model_args.use_emotion_consistency_loss = (
                self.config.model.use_emotion_consistency_loss
            )
        if not hasattr(model_config.model_args, 'emotion_embedding_file'):
            model_config.model_args.emotion_embedding_file = str(
                self.config.paths.emotion_embeddings_file
            )

        # Ensure SER model name is propagated so loss and embedding code align
        if not hasattr(model_config.model_args, 'ser_model_name'):
            model_config.model_args.ser_model_name = (
                self.config.model.ser_model_name
            )

        return model_config

    def _init_model(self, model_config):
        # Set VECL-specific attributes before model initialization
        model_config.model_args.use_emotion_consistency_loss = getattr(
            model_config, 'use_emotion_consistency_loss', False
        )
        model_config.model_args.emotion_embedding_file = getattr(
            model_config, 'emotion_embedding_file', None
        )

        # Ensure emotion_embedding_dim exists
        if not hasattr(model_config.model_args, 'emotion_embedding_dim'):
            model_config.model_args.emotion_embedding_dim = 768

        return Vecl.init_from_config(model_config)[0]

    def _init_model_for_inference(self, cfg):
        return Vecl.init_from_config(cfg)[0]

    def _get_inference_checkpoint_path(self) -> Path:
        return self.config.paths.restore_path


class YourTTSLoader(ModelLoader):
    """Loader for YourTTS models."""

    def _patch_config_for_training(self, model_config):
        # YourTTS doesn't need special config patches
        return model_config

    def _init_model(self, model_config):
        return setup_model(model_config)

    def _init_model_for_inference(self, cfg):
        return setup_model(cfg)

    def _get_inference_checkpoint_path(self) -> Path:
        return self.config.paths.pretrained_checkpoint_dir / 'best_model.pth'


def get_model_loader(config: AppConfig) -> ModelLoader:
    """Get the appropriate model loader based on config."""
    if config.model.type == 'vecl':
        return VeclLoader(config)
    elif config.model.type == 'yourtts':
        return YourTTSLoader(config)
    else:
        raise ValueError(f'Unsupported model type: {config.model.type}')


# Public API - these are the functions the training script uses
def load_model_for_training(config: AppConfig, dataset_configs: list):
    """Load model for training."""
    loader = get_model_loader(config)
    return loader.load_for_training(dataset_configs)


def load_model_for_inference(config: AppConfig, device: torch.device):
    """Load model for inference."""
    loader = get_model_loader(config)
    return loader.load_for_inference(device)
