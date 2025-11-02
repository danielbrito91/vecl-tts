from dataclasses import dataclass, field
from typing import List

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsArgs, VitsConfig

YOURTTS_CHARACTERS = CharactersConfig(
    characters_class='TTS.tts.models.vits.VitsCharacters',
    pad='_',
    eos='&',
    bos='*',
    blank=None,
    characters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00a1\u00a3\u00b7\u00b8\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c7\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d9\u00da\u00db\u00dc\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u0101\u0104\u0105\u0106\u0107\u010b\u0119\u0141\u0142\u0143\u0144\u0152\u0153\u015a\u015b\u0161\u0178\u0179\u017a\u017b\u017c\u020e\u04e7\u05c2\u1b20',
    punctuations="\u2014!'(),-.:;?\u00bf ",
    phonemes="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b'\u0303' ",
    is_unique=True,
    is_sorted=True,
)


@dataclass
class VeclArgs(VitsArgs):
    """
    Model arguments specific to VECL-TTS, inheriting from VitsArgs.
    """

    freeze_encoder: bool = True
    use_d_vector_file: bool = True
    emotion_embedding_file: str = None
    emotion_embedding_dim: int = 768
    d_vector_dim: int = 512
    use_emotion_consistency_loss: bool = True
    emotion_consistency_loss_alpha: float = 9.0

    # Language management parameters
    use_language_embedding: bool = True
    language_ids_file: str = None
    embedded_language_dim: int = 4


@dataclass
class VeclConfig(VitsConfig):
    """
    Main configuration for training a VECL-TTS model.
    """

    model_args: VeclArgs = field(default_factory=VeclArgs)
    characters: CharactersConfig = field(
        default_factory=lambda: YOURTTS_CHARACTERS
    )
    epochs: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 32
    eval_batch_size: int = 16
    num_loader_workers: int = 4
    save_step: int = 1000
    use_weighted_sampler: bool = True
    weighted_sampler_attrs: dict = field(
        default_factory=lambda: {'language': 1.0}
    )
    weighted_sampler_multipliers: dict = field(default_factory=dict)
    output_path: str = 'output'
    dashboard_logger: str = 'wandb'
    project_name: str = 'vecl-tts-finetune'
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [])
    max_text_len: int = 250
    max_audio_len: int = 22050 * 20

    # Loss weights overrides / additions
    speaker_encoder_loss_alpha: float = 9.0

    # Language management configuration
    use_language_weighted_sampler: bool = True
    language_weighted_sampler_alpha: float = 1.0

    def __post_init__(self):
        self.model_args.num_chars = len(self.characters)
        if self.weighted_sampler_multipliers is None:
            self.weighted_sampler_multipliers = {}

        # Set language-related parameters in model_args
        self.model_args.num_languages = self.model_args.num_languages
        self.num_languages = self.model_args.num_languages
