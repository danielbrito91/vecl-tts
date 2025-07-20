from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import nn
from TTS.tts.layers.losses import VitsDiscriminatorLoss
from TTS.tts.models.vits import Vits
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from vecl.models.config import VeclArgs, VeclConfig
from vecl.models.layers import EmotionProj
from vecl.models.loss import VeclGeneratorLoss


class Vecl(Vits):
    """
    The complete Vecl model with all necessary overrides for emotion/speaker fusion and data loading.
    """

    def __init__(
        self,
        config: VeclConfig,
        ap: 'AudioProcessor',
        tokenizer,
        speaker_manager,
        language_manager=None,
    ):
        super().__init__(
            config, ap, tokenizer, speaker_manager, language_manager
        )

        self.emotion_proj = None
        if (
            config.model_args.emotion_embedding_dim
            != config.model_args.d_vector_dim
        ):
            self.emotion_proj = EmotionProj(
                config.model_args.emotion_embedding_dim,
                config.model_args.d_vector_dim,
            )

    def _get_speaker_ids(self, batch: dict) -> torch.Tensor:
        """Get speaker IDs from batch."""
        speaker_ids = None
        if (
            self.speaker_manager is not None
            and self.speaker_manager.name_to_id
            and self.args.use_speaker_embedding
        ):
            speaker_ids = [
                self.speaker_manager.name_to_id[sn]
                for sn in batch['speaker_names']
            ]
            speaker_ids = torch.LongTensor(speaker_ids)
        return speaker_ids

    def _get_language_ids(self, batch: dict) -> torch.Tensor:
        """Get language IDs from batch."""
        language_ids = None
        if (
            self.language_manager is not None
            and self.language_manager.name_to_id
            and self.args.use_language_embedding
            and 'language_names' in batch
        ):
            try:
                language_ids = []
                for ln in batch['language_names']:
                    if ln in self.language_manager.name_to_id:
                        language_ids.append(
                            self.language_manager.name_to_id[ln]
                        )
                    else:
                        print(
                            f"⚠️ Language '{ln}' not found in language manager. Available: {list(self.language_manager.name_to_id.keys())}"
                        )
                        # Use the first available language as fallback
                        if self.language_manager.name_to_id:
                            fallback_lang = list(
                                self.language_manager.name_to_id.keys()
                            )[0]
                            language_ids.append(
                                self.language_manager.name_to_id[fallback_lang]
                            )
                            print(
                                f'   Using fallback language: {fallback_lang}'
                            )
                        else:
                            language_ids.append(0)  # Default fallback

                if language_ids:
                    language_ids = torch.LongTensor(language_ids)
            except Exception as e:
                print(f'⚠️ Error processing language IDs: {e}')
                language_ids = None
        return language_ids

    def _get_d_vectors(self, batch: dict) -> torch.Tensor:
        d_vectors = None

        if (
            self.args.use_d_vector_file
            and hasattr(self.speaker_manager, 'embeddings')
            and self.speaker_manager.embeddings
        ):
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors_list = []

            for name in batch['audio_unique_names']:
                # More comprehensive candidate key generation
                base_name = name.split('/')[-1]  # Get filename part
                name_no_ext = (
                    base_name.split('.')[0] if '.' in base_name else base_name
                )

                candidate_keys = [
                    # Original variations
                    name,
                    f'{name}.wav',
                    name.replace('-', '_'),
                    f'{name.replace("-", "_")}.wav',
                    name.replace('#audio/', '#audio_'),
                    f'{name.replace("#audio/", "#audio_")}.wav',
                    # Additional variations with just filename
                    base_name,
                    f'{base_name}.wav',
                    base_name.replace('-', '_'),
                    f'{base_name.replace("-", "_")}.wav',
                    # Variations without extension
                    name_no_ext,
                    name_no_ext.replace('-', '_'),
                    name_no_ext.replace('_', '-'),
                    # Common prefix patterns
                    f'audio/{name}',
                    f'audio/{base_name}',
                    f'#audio/{name}',
                    f'#audio/{base_name}',
                    # Try with different extensions
                    f'{name_no_ext}.flac',
                    f'{name_no_ext}.mp3',
                ]

                # Remove duplicates while preserving order
                seen = set()
                candidate_keys = [
                    x for x in candidate_keys if not (x in seen or seen.add(x))
                ]

                # Find match
                matched_key = next(
                    (ck for ck in candidate_keys if ck in d_vector_mapping),
                    None,
                )

                if matched_key is not None:
                    embedding_data = d_vector_mapping[matched_key]
                    if (
                        isinstance(embedding_data, dict)
                        and 'embedding' in embedding_data
                    ):
                        d_vectors_list.append(embedding_data['embedding'])
                    else:
                        d_vectors_list.append(embedding_data)

            if len(d_vectors_list) == len(batch['audio_unique_names']):
                d_vectors = torch.FloatTensor(d_vectors_list)

        return d_vectors

    def _fuse_emotion_embeddings(
        self, emotion_embeddings: torch.Tensor, d_vectors: torch.Tensor
    ) -> torch.Tensor:
        if emotion_embeddings.dim() == 3:
            emotion_embeddings = emotion_embeddings.squeeze(1)
        proj_device = next(self.emotion_proj.parameters()).device
        if emotion_embeddings.device != proj_device:
            emotion_embeddings = emotion_embeddings.to(proj_device)
        if d_vectors.device != proj_device:
            d_vectors = d_vectors.to(proj_device)
        projected_emo = self.emotion_proj(emotion_embeddings)
        d_vectors = torch.nn.functional.normalize(d_vectors + projected_emo)
        return d_vectors

    def format_batch(self, batch: dict) -> dict:
        """
        A complete and self-contained format_batch method.
        It handles speaker_ids, language_ids, and d_vectors correctly
        without calling the faulty parent method.
        """
        speaker_ids = self._get_speaker_ids(batch)
        language_ids = self._get_language_ids(batch)
        d_vectors = self._get_d_vectors(batch)
        emotion_embeddings = batch.get('emotion_embeddings', None)

        if (
            self.emotion_proj is not None
            and emotion_embeddings is not None
            and d_vectors is not None
        ):
            d_vectors = self._fuse_emotion_embeddings(
                emotion_embeddings, d_vectors
            )

        # The Coqui TTS Trainer has a check that requires speaker_ids to be
        # present if a speaker_manager is available, even if we are using d_vectors.
        # To bypass this, we create a placeholder tensor for speaker_ids if it's None
        # but d_vectors are present. This prevents a crash in the trainer.
        if speaker_ids is None and d_vectors is not None:
            # Create a tensor of zeros with the same batch size and device as d_vectors
            speaker_ids = torch.zeros(
                d_vectors.size(0), dtype=torch.long, device=d_vectors.device
            )

        batch['d_vectors'] = d_vectors
        batch['speaker_ids'] = speaker_ids
        batch['language_ids'] = language_ids
        return batch

    def forward(self, x, x_lengths, y, y_lengths, waveform, aux_input=None):  # noqa: D401,E501
        """Keep signature-compatible forward that simply calls parent since fusion moved to `format_batch`."""
        return super().forward(
            x, x_lengths, y, y_lengths, waveform, aux_input=aux_input
        )

    def get_criterion(self):
        """Return the custom loss functions for VECL-TTS."""
        return [
            VitsDiscriminatorLoss(self.config),
            VeclGeneratorLoss(self.config),
        ]

    def train_step(
        self, batch: dict, criterion: nn.Module, optimizer_idx: int
    ):
        # Spec and Mel are computed on the device for more efficient memory usage.
        batch = self.format_batch_on_device(batch)

        if optimizer_idx == 0:
            # The discriminator step in VITS computes generator outputs and caches them.
            # We then pass these to the discriminator.
            return super().train_step(batch, criterion, optimizer_idx)

        if optimizer_idx == 1:
            # The generator step uses the cached outputs from the discriminator pass.
            outputs = self.model_outputs_cache
            from TTS.tts.models.vits import wav_to_mel
            from TTS.tts.utils.helpers import segment

            with torch.autocast('cuda', enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = int(
                        self.spec_segment_size * self.interpolate_factor
                    )
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    batch['mel'].float(),
                    outputs['slice_ids'],
                    spec_segment_size,
                    pad_short=True,
                )
                mel_slice_hat = wav_to_mel(
                    y=outputs['model_outputs'].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                outputs['model_outputs'], outputs['waveform_seg']
            )
            with torch.autocast('cuda', enabled=False):
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=outputs['z_p'].float(),
                    logs_q=outputs['logs_q'].float(),
                    m_p=outputs['m_p'].float(),
                    logs_p=outputs['logs_p'].float(),
                    z_len=batch['spec_lens'],
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=outputs['loss_duration'],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=outputs.get('gt_spk_emb'),
                    syn_spk_emb=outputs.get('syn_spk_emb'),
                    generated_wav=outputs['model_outputs'],
                    ref_emotion_embeddings=batch.get('emotion_embeddings'),
                    sample_rate=self.config.audio.sample_rate,
                )
            return outputs, loss_dict
        raise ValueError(' [!] Unexpected `optimizer_idx`.')

    @staticmethod
    def init_from_config(
        config: VeclConfig,
        samples: Union[List[List], List[Dict]] = None,
    ):
        """Initiate model from config"""
        # Ensure model_args is the correct type, not a dict
        if isinstance(config.model_args, dict):
            config.model_args = VeclArgs(**config.model_args)

        ap = AudioProcessor.init_from_config(config)
        tokenizer, tokenizer_config = TTSTokenizer.init_from_config(config)

        # SpeakerManager will find the d_vector_file in the config object
        speaker_manager = SpeakerManager.init_from_config(
            config, samples=samples
        )

        # Initialize language manager with proper configuration
        language_manager = None
        if getattr(config.model_args, 'use_language_embedding', False):
            # Check if language_ids_file exists and is configured
            language_ids_file = getattr(
                config.model_args, 'language_ids_file', None
            )
            if language_ids_file and Path(language_ids_file).exists():
                print(
                    f'🌍 Initializing language manager from: {language_ids_file}'
                )

                # CRITICAL FIX: LanguageManager expects language_ids_file at the top level of config
                # not just in model_args. We need to set it at both levels.
                config.language_ids_file = str(language_ids_file)
                config.model_args.language_ids_file = str(language_ids_file)

                # Also set num_languages from the JSON file
                import json

                with open(language_ids_file, 'r') as f:
                    language_ids = json.load(f)
                config.model_args.num_languages = len(language_ids)

                language_manager = LanguageManager.init_from_config(config)

                # Update config with actual language count from the loaded manager
                if language_manager and hasattr(
                    language_manager, 'name_to_id'
                ):
                    actual_lang_count = len(language_manager.name_to_id)
                    config.model_args.num_languages = actual_lang_count
                    print(
                        f'✅ Language manager initialized with {actual_lang_count} languages: {list(language_manager.name_to_id.keys())}'
                    )
                else:
                    print('⚠️ Language manager initialized but appears empty')
            else:
                print(
                    f'⚠️ Language embeddings enabled but language_ids_file not found: {language_ids_file}'
                )
                print('   Creating empty language manager for compatibility')
                language_manager = LanguageManager()
        else:
            print(
                '🌍 Language embeddings disabled - no language manager created'
            )

        model = Vecl(config, ap, tokenizer, speaker_manager, language_manager)
        return model, config

    def get_sampler(self, config, dataset, num_gpus=1, is_eval=False):  # noqa: D401,E501
        """Return a sampler but guarantee required config dicts are valid."""
        if getattr(config, 'weighted_sampler_multipliers', None) is None:
            config.weighted_sampler_multipliers = {}

        if getattr(config, 'weighted_sampler_attrs', None) is None:
            config.weighted_sampler_attrs = {}

        if len(dataset) < getattr(config, 'batch_size', 1):
            config.batch_size = len(dataset)
            config.eval_batch_size = len(dataset)
            config.use_weighted_sampler = False

        return super().get_sampler(
            config, dataset, num_gpus=num_gpus, is_eval=is_eval
        )

    def get_data_loader(
        self,
        config,
        assets,
        is_eval,
        samples,
        verbose,
        num_gpus,
        rank=None,
    ):
        """Wrap parent implementation but enlarge `max_text_len` for debug."""
        original_max_text_len = config.max_text_len

        try:
            if len(samples) < 8:
                config.max_text_len = 10_000

            from vecl.data.vecl_dataset import (
                VeclDataset,
            )

            dataset = VeclDataset(
                model_args=self.args,
                samples=samples,
                batch_group_size=0
                if is_eval
                else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                emotion_embedding_file=getattr(
                    self.args, 'emotion_embedding_file', None
                ),
            )

            dataset.preprocess_samples()

            if num_gpus > 1:
                import torch.distributed as dist

                dist.barrier()

            sampler = self.get_sampler(
                config, dataset, num_gpus, is_eval=is_eval
            )

            if sampler is None:
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size
                    if is_eval
                    else config.batch_size,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    num_workers=config.num_eval_loader_workers
                    if is_eval
                    else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                if num_gpus > 1:
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=config.eval_batch_size
                        if is_eval
                        else config.batch_size,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers
                        if is_eval
                        else config.num_loader_workers,
                        pin_memory=False,
                    )
                else:
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers
                        if is_eval
                        else config.num_loader_workers,
                        pin_memory=False,
                    )

            return loader
        finally:
            config.max_text_len = original_max_text_len

    def get_aux_input_from_test_sentences(self, sentence_info):  # noqa: D401,E501
        """Robust version that trims whitespace and tolerates missing speakers."""
        text, utt_id, speaker_name, language = sentence_info

        speaker_name = (
            speaker_name.strip()
            if isinstance(speaker_name, str)
            else speaker_name
        )

        aux = {
            'text': text,
            'd_vector': None,
            'speaker_id': None,
            'language_id': None,
            'd_vectors': None,
            'speaker_ids': None,
            'language_ids': None,
            'style_wav': None,
            'style_text': None,
        }

        if (
            self.language_manager is not None
            and language in self.language_manager.name_to_id
        ):
            lang_id = self.language_manager.name_to_id[language]
            aux['language_ids'] = torch.LongTensor([[lang_id]])
            aux['language_id'] = torch.LongTensor([lang_id])

        if self.speaker_manager is not None and speaker_name:
            try:
                dvec = self.speaker_manager.get_mean_embedding(
                    speaker_name, num_samples=None, randomize=False
                )
                aux['d_vectors'] = dvec.unsqueeze(0)
                aux['d_vector'] = dvec

                if speaker_name in self.speaker_manager.name_to_id:
                    spk_id = self.speaker_manager.name_to_id[speaker_name]
                    aux['speaker_ids'] = torch.LongTensor([[spk_id]])
                    aux['speaker_id'] = torch.LongTensor([spk_id])
            except KeyError:
                print(
                    f" [!] Speaker '{speaker_name}' not found in embeddings; proceeding without d_vector."
                )

        if (
            aux['d_vectors'] is None
            and getattr(self.args, 'use_d_vector_file', False)
            and getattr(self.args, 'd_vector_dim', 0) > 0
        ):
            dim = self.args.d_vector_dim
            zeros = torch.zeros(dim)
            aux['d_vector'] = zeros
            aux['d_vectors'] = zeros.unsqueeze(0)

        return aux
