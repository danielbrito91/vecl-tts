import json
from pathlib import Path
from typing import List

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import (
    CharactersConfig,
    VitsArgs,
    VitsAudioConfig,
)
from TTS.utils.download import download_url

SPEAKER_ENCODER_CONFIG_URL = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json'
SPEAKER_ENCODER_CHECKPOINT_PATH = 'https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar'


def load_config(
    checkpoint_dir: str,
    speaker_encoder_model_dir: Path,
    output_path: str,
    dataset_configs: List[BaseDatasetConfig],
    embeddings_file: str,
    # Training params
    batch_size: int,
    eval_batch_size: int,
    num_loader_workers: int,
    epochs: int,
    learning_rate: float,
    save_step: int,
    max_text_len: int,
    # Audio params
    sample_rate: int,
    max_audio_len_seconds: int,
    # Logging params
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str,
    wandb_entity: str,
    # Custom params
    use_original_lang_ids: bool = False,
) -> VitsConfig:
    """
    Loads a VitsConfig from a pre-trained model's directory and updates it
    with parameters for fine-tuning on a new dataset.
    """
    print('\n>>> Configuring VITS model for fine-tuning...')

    se_config_path = speaker_encoder_model_dir / 'config_se.json'
    if not se_config_path.exists():
        print(
            f'Downloading speaker encoder config from {SPEAKER_ENCODER_CONFIG_URL} to {speaker_encoder_model_dir}'
        )
        speaker_encoder_model_dir.mkdir(parents=True, exist_ok=True)
        download_url(
            SPEAKER_ENCODER_CONFIG_URL, str(speaker_encoder_model_dir)
        )

    se_model_path = (
        speaker_encoder_model_dir / Path(SPEAKER_ENCODER_CHECKPOINT_PATH).name
    )
    if not se_model_path.exists():
        print(
            f'Downloading speaker encoder model from {SPEAKER_ENCODER_CHECKPOINT_PATH} to {speaker_encoder_model_dir}'
        )
        speaker_encoder_model_dir.mkdir(parents=True, exist_ok=True)
        download_url(
            SPEAKER_ENCODER_CHECKPOINT_PATH, str(speaker_encoder_model_dir)
        )

    config_path = Path(checkpoint_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found at {config_path}.')

    # Load the base config from the checkpoint as a dictionary
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Manually create VitsArgs from the dictionary to avoid deserialization issues.
    # This prevents the `UserWarning` and keeps the pre-trained model's architecture.
    model_args_from_json = config_dict.get('model_args', {})
    model_args = VitsArgs()
    for key, value in model_args_from_json.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)

    # Create the main config object and assign the manually created model_args
    config = VitsConfig(model_args=model_args)

    # Now, update the rest of the config fields from the loaded dictionary
    for key, value in config_dict.items():
        if key != 'model_args' and hasattr(config, key):
            setattr(config, key, value)

    language_ids_file = None
    num_languages = 0
    if use_original_lang_ids:
        print('  > Using original language IDs from pretrained model.')
        language_ids_file = Path(checkpoint_dir) / 'language_ids.json'
        if not language_ids_file.exists():
            raise FileNotFoundError(
                f'language_ids.json not found in checkpoint_dir: {checkpoint_dir}'
            )
        with open(language_ids_file, 'r') as f:
            language_ids_dict = json.load(f)
        num_languages = len(language_ids_dict)
        print(f'  > Number of languages: {num_languages}')
    elif config.datasets:
        # if a new dataset is provided, we need to extract the languages
        # and create a new language_ids.json file
        languages = {lang for dc in dataset_configs for lang in dc.languages}
        num_languages = len(languages)
        config.languages = list(languages)

    correct_lang_id_path = (
        str(language_ids_file) if language_ids_file else None
    )

    audio_config = VitsAudioConfig(
        sample_rate=sample_rate,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )
    model_args = VitsArgs(
        spec_segment_size=62,
        hidden_channels=192,
        hidden_channels_ffn_text_encoder=768,
        num_heads_text_encoder=2,
        num_layers_text_encoder=10,
        kernel_size_text_encoder=3,
        dropout_p_text_encoder=0.1,
        d_vector_file=[str(embeddings_file)],
        use_d_vector_file=True,
        d_vector_dim=512,
        speaker_encoder_model_path=str(se_model_path),
        speaker_encoder_config_path=str(se_config_path),
        resblock_type_decoder='2',  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
        # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
        use_speaker_encoder_as_loss=False,
        # Useful parameters to enable multilingual training
        use_language_embedding=True,
        embedded_language_dim=4,
        num_languages=num_languages,
        language_ids_file=correct_lang_id_path,
    )
    config.model_args = model_args
    config.output_path = output_path
    config.run_name = wandb_run_name
    config.project_name = wandb_project
    config.epochs = epochs
    config.lr = learning_rate
    config.run_description = """
            - YourTTS trained using CML-TTS and LibriTTS datasets
        """
    config.dashboard_logger = 'wandb' if use_wandb else 'tensorboard'
    config.logger_uri = wandb_entity if use_wandb else None
    config.audio = audio_config
    config.batch_size = batch_size
    config.batch_group_size = 48
    config.eval_batch_size = eval_batch_size
    config.num_loader_workers = num_loader_workers
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 1000
    config.save_step = save_step
    config.max_text_len = max_text_len
    config.save_n_checkpoints = 2
    config.save_checkpoints = True
    config.target_loss = 'loss_1'
    config.print_eval = False
    config.use_phonemes = False
    config.phonemizer = 'espeak'
    config.phoneme_language = 'en'
    config.compute_input_seq_cache = True
    config.add_blank = True
    config.text_cleaner = 'multilingual_cleaners'
    config.characters = CharactersConfig(
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
    config.phoneme_cache_path = None
    config.precompute_num_workers = 12
    config.start_by_longest = True
    config.datasets = dataset_configs
    config.cudnn_benchmark = False
    config.max_audio_len = sample_rate * max_audio_len_seconds
    config.mixed_precision = False
    config.test_sentences = [
        [
            'Voc\u00ea ter\u00e1 a vista do topo da montanha que voc\u00ea escalar.',
            '9351',
            None,
            'pt-br',
        ],
        [
            'Quando voc\u00ea n\u00e3o corre nenhum risco, voc\u00ea arrisca tudo.',
            '12249',
            None,
            'pt-br',
        ],
        [
            'S\u00e3o necess\u00e1rios muitos anos de trabalho para ter sucesso da noite para o dia.',
            '2961',
            None,
            'pt-br',
        ],
        [
            "You'll have the view of the top of the mountain that you climb.",
            'LTTS_6574',
            None,
            'en',
        ],
        [
            'When you don\u2019t take any risks, you risk everything.',
            'LTTS_6206',
            None,
            'en',
        ],
        [
            'Are necessary too many years of work to succeed overnight.',
            'LTTS_5717',
            None,
            'en',
        ],
        [
            'Je hebt uitzicht op de top van de berg die je beklimt.',
            '960',
            None,
            'du',
        ],
        [
            'Als je geen risico neemt, riskeer je alles.',
            '2450',
            None,
            'du',
        ],
        [
            'Zijn te veel jaren werk nodig om van de ene op de andere dag te slagen.',
            '10984',
            None,
            'du',
        ],
        [
            'Vous aurez la vue sur le sommet de la montagne que vous gravirez.',
            '6381',
            None,
            'fr',
        ],
        [
            'Quand tu ne prends aucun risque, tu risques tout.',
            '2825',
            None,
            'fr',
        ],
        [
            "Sont n\u00e9cessaires trop d'ann\u00e9es de travail pour r\u00e9ussir du jour au lendemain.",
            '1844',
            None,
            'fr',
        ],
        [
            'Sie haben die Aussicht auf die Spitze des Berges, den Sie erklimmen.',
            '2314',
            None,
            'ge',
        ],
        ['Wer nichts riskiert, riskiert alles.', '7483', None, 'ge'],
        [
            'Es sind zu viele Jahre Arbeit notwendig, um \u00fcber Nacht erfolgreich zu sein.',
            '12461',
            None,
            'ge',
        ],
        [
            'Avrai la vista della cima della montagna che sali.',
            '4998',
            None,
            'it',
        ],
        [
            'Quando non corri alcun rischio, rischi tutto.',
            '6744',
            None,
            'it',
        ],
        [
            'Are necessary too many years of work to succeed overnight.',
            '1157',
            None,
            'it',
        ],
        [
            'B\u0119dziesz mie\u0107 widok na szczyt g\u00f3ry, na kt\u00f3r\u0105 si\u0119 wspinasz.',
            '7014',
            None,
            'pl',
        ],
        [
            'Kiedy nie podejmujesz \u017cadnego ryzyka, ryzykujesz wszystko.',
            '2933',
            None,
            'pl',
        ],
        [
            'Potrzeba zbyt wielu lat pracy, by odnie\u015b\u0107 sukces z dnia na dzie\u0144.',
            '11634',
            None,
            'pl',
        ],
        [
            'Tendr\u00e1s la vista de la cima de la monta\u00f1a que subes.',
            '433',
            None,
            'sp',
        ],
        [
            'Cuando no se arriesga, se arriesga todo.',
            '3518',
            None,
            'sp',
        ],
        [
            'Son necesarios demasiados a\u00f1os de trabajo para tener \u00e9xito de la noche a la ma\u00f1ana.',
            '3217',
            None,
            'sp',
        ],
    ]

    # Update VitsArgs with fine-tuning parameters, preserving the rest
    config.model_args.d_vector_file = [str(embeddings_file)]
    config.model_args.use_d_vector_file = True
    config.model_args.speaker_encoder_model_path = str(se_model_path)
    config.model_args.speaker_encoder_config_path = str(se_config_path)
    config.model_args.num_languages = num_languages
    config.model_args.language_ids_file = correct_lang_id_path

    # ALSO update the top-level config attribute that LanguageManager uses
    config.language_ids_file = correct_lang_id_path

    # Update top-level config with fine-tuning parameters
    config.output_path = output_path
    config.run_name = wandb_run_name
    config.project_name = wandb_project
    config.epochs = epochs
    config.lr = learning_rate
    config.run_description = """
            - YourTTS trained using CML-TTS and LibriTTS datasets
        """
    config.dashboard_logger = 'wandb' if use_wandb else 'tensorboard'
    config.logger_uri = wandb_entity if use_wandb else None
    config.audio = VitsAudioConfig(
        sample_rate=sample_rate,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )
    config.batch_size = batch_size
    config.batch_group_size = 48
    config.eval_batch_size = eval_batch_size
    config.num_loader_workers = num_loader_workers
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 1000
    config.save_step = save_step
    config.max_text_len = max_text_len
    config.save_n_checkpoints = 2
    config.save_checkpoints = True
    config.target_loss = 'loss_1'
    config.print_eval = False
    config.use_phonemes = False
    config.phonemizer = 'espeak'
    config.phoneme_language = 'en'
    config.compute_input_seq_cache = True
    config.add_blank = True
    config.text_cleaner = 'multilingual_cleaners'
    config.datasets = dataset_configs
    config.max_audio_len = sample_rate * max_audio_len_seconds

    print('  > ✅ VITS config loaded and updated for fine-tuning.')
    return config
