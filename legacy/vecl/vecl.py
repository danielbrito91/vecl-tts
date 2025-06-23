import torch
from coqpit import Coqpit
from torch import nn
from TTS.tts.layers.losses import (
    VitsDiscriminatorLoss,
)
from TTS.tts.models.vits import Vits
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from vecl.vecl.config import VeclConfig
from vecl.vecl.emotion_embedding import EmotionProj
from vecl.vecl.loss import VeclGeneratorLoss


class Vecl(Vits):
    """
    The complete Vecl model with all necessary overrides for emotion/speaker fusion and data loading.
    """

    def __init__(
        self,
        config: Coqpit,
        ap: 'AudioProcessor',
        tokenizer,
        speaker_manager,
        language_manager=None,
    ):
        # Pass all arguments to the parent Vits class
        super().__init__(
            config, ap, tokenizer, speaker_manager, language_manager
        )

        # Initialize the emotion projection layer
        self.emotion_proj = None
        if (
            config.model_args.emotion_embedding_dim
            != config.model_args.d_vector_dim
        ):
            self.emotion_proj = EmotionProj(
                config.model_args.emotion_embedding_dim,
                config.model_args.d_vector_dim,
            )

    def format_batch(self, batch: dict) -> dict:
        """
        A complete and self-contained format_batch method.
        It handles speaker_ids, language_ids, and d_vectors correctly
        without calling the faulty parent method.
        """
        # 1. Get speaker_ids if use_speaker_embedding is enabled
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

        # 2. Get language_ids if use_language_embedding is enabled
        language_ids = None
        if (
            self.language_manager is not None
            and self.language_manager.name_to_id
            and self.args.use_language_embedding
        ):
            language_ids = [
                self.language_manager.name_to_id[ln]
                for ln in batch['language_names']
            ]
            language_ids = torch.LongTensor(language_ids)

        # 3. Correctly load d-vectors from SpeakerManager.embeddings (Coqui naming)
        d_vectors = None
        if (
            self.args.use_d_vector_file
            and hasattr(self.speaker_manager, 'embeddings')
            and self.speaker_manager.embeddings
        ):
            # The speaker embeddings dictionary lives under `speaker_manager.embeddings` in Coqui-TTS
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors_list = []
            for name in batch['audio_unique_names']:
                candidate_keys = [
                    name,  # original
                    f'{name}.wav',  # add extension
                    name.replace('-', '_'),  # replace dashes with underscores
                    f'{name.replace("-", "_")}.wav',  # combo
                    name.replace('#audio/', '#audio_'),  # slash -> underscore
                    f'{name.replace("#audio/", "#audio_")}.wav',
                ]

                matched_key = next(
                    (ck for ck in candidate_keys if ck in d_vector_mapping),
                    None,
                )

                if matched_key is not None:
                    embedding_data = d_vector_mapping[matched_key]
                    # dict or tensor
                    if (
                        isinstance(embedding_data, dict)
                        and 'embedding' in embedding_data
                    ):
                        d_vectors_list.append(embedding_data['embedding'])
                    else:
                        d_vectors_list.append(embedding_data)
                else:
                    print(
                        f" [!] Warning: 'audio_unique_name' not found in d_vector mapping: {name}"
                    )

            if len(d_vectors_list) == len(batch['audio_unique_names']):
                d_vectors = torch.FloatTensor(d_vectors_list)
            else:
                print(
                    ' [!] D-vectors not loaded for the batch because some keys were missing.'
                )

        # 4. Fuse emotion embeddings -> d_vectors (projection + L2-norm)
        emotion_embeddings = batch.get('emotion_embeddings', None)
        if (
            self.emotion_proj is not None
            and emotion_embeddings is not None
            and d_vectors is not None
        ):
            if emotion_embeddings.dim() == 3:
                emotion_embeddings = emotion_embeddings.squeeze(1)
            # Ensure tensors are on the same device as the projection layer
            proj_device = next(self.emotion_proj.parameters()).device
            if emotion_embeddings.device != proj_device:
                emotion_embeddings = emotion_embeddings.to(proj_device)
            if d_vectors.device != proj_device:
                d_vectors = d_vectors.to(proj_device)
            projected_emo = self.emotion_proj(emotion_embeddings)
            d_vectors = torch.nn.functional.normalize(
                d_vectors + projected_emo
            )

        # 5. Update the batch dictionary with all computed values
        batch['d_vectors'] = d_vectors
        batch['speaker_ids'] = speaker_ids
        batch['language_ids'] = language_ids
        return batch

    # Keep signature-compatible forward that simply calls parent since fusion
    # moved to `format_batch`.
    def forward(self, x, x_lengths, y, y_lengths, waveform, aux_input=None):  # noqa: D401,E501
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
        """Override the train_step to pass additional arguments to the loss function."""
        # ------------------------------------------------------------------
        # SAFETY PATCH — ensure `.forward` has the expected signature.
        # After calling `export_onnx()` Coqui-TTS temporarily replaces
        # `self.forward` with a 3-arg inference stub.  If a dev accidentally
        # calls `export_onnx()` in the same Python process and then tries to
        # resume training, `Vits.train_step` will invoke `forward()` with six
        # positional arguments and raise a `TypeError`.  Here we detect that
        # situation and restore the original implementation on-the-fly.
        # ------------------------------------------------------------------
        if self.forward.__code__.co_argcount < 6:  # monkey-patched version
            # Re-bind the class method to this instance.
            self.forward = Vecl.forward.__get__(self)

        if optimizer_idx == 0:
            # The discriminator step is unchanged
            return super().train_step(batch, criterion, optimizer_idx)

        if optimizer_idx == 1:
            # ------------------------------------------------------------------
            # Generator (optimizer_idx == 1)
            # ------------------------------------------------------------------
            # Use the cached tensors from the discriminator pass.
            outputs = (
                self.model_outputs_cache
            )  # calculated when optimizer_idx==0

            # ----- Compute mel segments (ground-truth and generated) ----------
            # This mirrors the upstream VITS implementation which relies on
            # `segment` + `wav_to_mel` utilities that work directly with torch
            # tensors – avoiding the Librosa/numpy path that tripped the
            # previous error.

            from TTS.tts.utils.helpers import (
                segment,  # local import to avoid circular issues
            )
            from TTS.utils.audio.torch_transforms import wav_to_mel

            with torch.autocast('cuda', enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = int(
                        self.spec_segment_size * self.interpolate_factor
                    )
                else:
                    spec_segment_size = self.spec_segment_size

                # Ground-truth mel slice (target)
                mel_slice = segment(
                    batch['mel'].float(),
                    outputs['slice_ids'],
                    spec_segment_size,
                    pad_short=True,
                )

                # Predicted mel slice (hat)
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

            # ----- Discriminator scores ---------------------------------------
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                outputs['model_outputs'], outputs['waveform_seg']
            )

            # ----- Compute VECL generator loss --------------------------------
            with torch.autocast('cuda', enabled=False):
                loss_dict = criterion[
                    optimizer_idx
                ](
                    mel_slice_hat=mel_slice.float(),  # follow upstream arg order
                    mel_slice=mel_slice_hat.float(),  # ^ see VITS implementation
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
                    gt_spk_emb=outputs['gt_spk_emb'],
                    syn_spk_emb=outputs['syn_spk_emb'],
                    generated_wav=outputs['model_outputs'],
                    ref_emotion_embeddings=batch.get(
                        'emotion_embeddings', None
                    ),
                    sample_rate=self.config.audio.sample_rate,
                )

            return outputs, loss_dict

        raise ValueError(' [!] Unexpected `optimizer_idx`.')

    @staticmethod
    def init_from_config(config: 'VeclConfig', samples: list = None):
        """Initiate a Vecl model and its components from a config object."""
        # This method is correct and remains the same
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)

        # --- Patch: TTSTokenizer.init_from_config may return a shallow copy of the
        # config that drops custom attributes that exist only in our subclass.
        # Guarantee the sampler-related attributes are present to avoid
        # AttributeErrors downstream.
        if (
            not hasattr(new_config, 'weighted_sampler_multipliers')
            or new_config.weighted_sampler_multipliers is None
        ):
            new_config.weighted_sampler_multipliers = {}
        if (
            not hasattr(new_config, 'weighted_sampler_attrs')
            or new_config.weighted_sampler_attrs is None
        ):
            new_config.weighted_sampler_attrs = {'language': 1.0}

        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path,
                config.model_args.speaker_encoder_config_path,
            )

        return Vecl(
            new_config, ap, tokenizer, speaker_manager, language_manager
        )

    # ------------------------------------------------------------------
    # Sampler override – make training robust to missing config fields
    # ------------------------------------------------------------------
    def get_sampler(self, config, dataset, num_gpus=1, is_eval=False):  # noqa: D401,E501
        """Return a sampler but guarantee required config dicts are valid.

        Coqui-TTS expects `config.weighted_sampler_multipliers` to be a dict.
        When the config is serialized/deserialized by Coqpit it can become
        `None`, leading to `AttributeError` in the base implementation.
        We defensively replace `None` with an empty dict before delegating
        to `Vits.get_sampler`.
        """

        if getattr(config, 'weighted_sampler_multipliers', None) is None:
            config.weighted_sampler_multipliers = {}

        if getattr(config, 'weighted_sampler_attrs', None) is None:
            config.weighted_sampler_attrs = {}

        # --- Mini-dataset safeguard -------------------------------------------------
        # If we are running a tiny local test (e.g., only a few samples in memory)
        # but keep the default batch_size=32, BucketBatchSampler with
        # `drop_last=True` will drop the incomplete batch, leaving the dataloader
        # empty and triggering an AssertionError.  Detect that situation and
        # down-scale the batch size or disable the weighted sampler so that at
        # least one batch is produced.

        if len(dataset) < getattr(config, 'batch_size', 1):
            # Reduce batch size to dataset length
            config.batch_size = len(dataset)
            config.eval_batch_size = len(dataset)
            # A tiny dataset does not need a weighted sampler.
            config.use_weighted_sampler = False

        return super().get_sampler(
            config, dataset, num_gpus=num_gpus, is_eval=is_eval
        )

    # ------------------------------------------------------------------
    # Data-loader override – relax text length limits for debug mini-runs
    # ------------------------------------------------------------------
    def get_data_loader(
        self,
        config,
        assets,
        is_eval,
        samples,
        verbose,
        num_gpus,
        rank=None,
    ):  # noqa: D401,E501
        """Wrap parent implementation but enlarge `max_text_len` when we are
        deliberately feeding a *very* small set of samples for quick local
        checks (like the 2-sample notebook sanity run).  This prevents
        `VitsDataset.__getitem__` from recursively discarding every sample and
        throwing an `IndexError`.
        """

        original_max_text_len = config.max_text_len

        try:
            # Detect mini-dataset: < 8 examples means user is probably running a
            # quick interactive check.  Heuristically relax the text length
            # constraint.
            if len(samples) < 8:
                config.max_text_len = 10_000  # effectively no limit

            # ------------------------------------------------------------------
            # Choose dataset class: use `VeclDataset` when an emotion embedding
            # file is provided, otherwise fall back to the base `VitsDataset`.
            # ------------------------------------------------------------------

            from vecl.vecl.dataset import (
                VeclDataset,  # always use the VECL-aware dataset
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

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # wait for all processes in DDP
            if num_gpus > 1:
                import torch.distributed as dist

                dist.barrier()

            # ----- Sampler logic (same as VITS) -----------------------------
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
            # Always restore original value so real training is unaffected.
            config.max_text_len = original_max_text_len

    def get_aux_input_from_test_sentences(self, sentence_info):  # noqa: D401,E501
        """Robust version that trims whitespace and tolerates missing speakers.

        sentence_info is a tuple like (text, utt_id, speaker_name, language).
        Upstream crashes if `speaker_name` is not found inside
        `SpeakerManager.embeddings_by_names`.  This happens in some CML test
        sentences where the ID in the CSV has trailing new-line chars.  We
        sanitise and fall back to *no* speaker conditioning if still missing.
        """

        import torch

        text, utt_id, speaker_name, language = sentence_info

        speaker_name = (
            speaker_name.strip()
            if isinstance(speaker_name, str)
            else speaker_name
        )

        aux = {
            'text': text,  # required by VITS.test_run()
            # singular keys used by VITS.test_run -> inference()
            'd_vector': None,
            'speaker_id': None,
            'language_id': None,
            # plural variants still used elsewhere in VECL overrides
            'd_vectors': None,
            'speaker_ids': None,
            'language_ids': None,
            'style_wav': None,
            'style_text': None,
        }

        # Language embedding --------------------------------------------------
        if (
            self.language_manager is not None
            and language in self.language_manager.name_to_id
        ):
            lang_id = self.language_manager.name_to_id[language]
            aux['language_ids'] = torch.LongTensor([[lang_id]])
            aux['language_id'] = torch.LongTensor([lang_id])

        # Speaker embedding ---------------------------------------------------
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

        # ------------------------------------------------------------------
        # Guarantee that HiFi-GAN conditioner always receives a tensor.
        # When the model was trained with d-vectors, ``self.waveform_decoder``
        # has ``cond_layer`` and will crash if ``g`` (speaker conditioning)
        # is None.  Provide a zero-vector with proper shape when we could
        # not retrieve any real embedding.
        #
        # NOTE: `Vits.inference` expects `d_vectors` (plural). The `synthesis`
        # helper passes `d_vector` (singular), but we populate `d_vectors`
        # here to be safe and ensure the correct key is used downstream.
        # The synthesis utility expects CPU tensors, so we create the fallback
        # tensor on CPU.
        # ------------------------------------------------------------------
        if (
            aux['d_vectors'] is None
            and getattr(self.args, 'use_d_vector_file', False)
            and getattr(self.args, 'd_vector_dim', 0) > 0
        ):
            dim = self.args.d_vector_dim
            zeros = torch.zeros(dim)  # on CPU
            aux['d_vector'] = zeros
            aux['d_vectors'] = zeros.unsqueeze(0)

        return aux
