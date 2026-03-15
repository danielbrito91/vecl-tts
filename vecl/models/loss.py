import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from TTS.config import Coqpit
from TTS.tts.layers.losses import VitsGeneratorLoss
from speechbrain.inference.interfaces import foreign_class


class VeclGeneratorLoss(VitsGeneratorLoss):
    """
    Generator loss for VECL-TTS, which efficiently integrates the emotion consistency logic.
    """

    def __init__(self, c: Coqpit):
        super().__init__(c)
        self.use_emotion_consistency_loss = getattr(
            getattr(c, 'model_args', c), 'use_emotion_consistency_loss', False
        )

        if self.use_emotion_consistency_loss:
            print(' > Using Emotion Consistency Loss.')
            self.emotion_consistency_loss_alpha = getattr(
                getattr(c, 'model_args', c),
                'emotion_consistency_loss_alpha',
                1.0,
            )
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            # Select SER backend based on config to match precomputed embeddings
            self.ser_model_name = getattr(
                getattr(c, 'model_args', c),
                'ser_model_name',
                'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
            )

            if str(self.ser_model_name).startswith('alefiury'):
                self.ser_backend = 'hf'
                self.emo_feat_ext = AutoFeatureExtractor.from_pretrained(
                    self.ser_model_name
                )
                self.emo_encoder = AutoModelForAudioClassification.from_pretrained(
                    self.ser_model_name
                )
                self.emo_encoder.eval()
                for param in self.emo_encoder.parameters():
                    param.requires_grad = False
                self.emo_encoder.to(self.device)
            elif str(self.ser_model_name).startswith('speechbrain'):
                self.ser_backend = 'speechbrain'
                self.sb_model = foreign_class(
                    source=self.ser_model_name,
                    pymodule_file='custom_interface.py',
                    classname='CustomEncoderWav2vec2Classifier',
                )
                self.sb_model.eval()
                self._ensure_sb_on_device(self.device)
            else:
                # Default to HF path if unknown
                self.ser_backend = 'hf'
                self.emo_feat_ext = AutoFeatureExtractor.from_pretrained(
                    self.ser_model_name
                )
                self.emo_encoder = AutoModelForAudioClassification.from_pretrained(
                    self.ser_model_name
                )
                self.emo_encoder.eval()
                for param in self.emo_encoder.parameters():
                    param.requires_grad = False
                self.emo_encoder.to(self.device)

    def _get_emotion_embedding_from_tensor(
        self, waveform_tensor: torch.Tensor, sample_rate: int
    ):
        """
        Computes emotion embedding from a waveform tensor in memory.
        Uses the configured SER backend to match precomputed embeddings.
        """
        waveform_tensor = waveform_tensor.to(self.device)

        # Ensure mono
        if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0)

        if self.ser_backend == 'hf':
            if sample_rate != self.emo_feat_ext.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.emo_feat_ext.sampling_rate,
                ).to(self.device)
                waveform_tensor = resampler(waveform_tensor)

            waveform_tensor = waveform_tensor.squeeze()
            inputs = self.emo_feat_ext(
                waveform_tensor,
                sampling_rate=self.emo_feat_ext.sampling_rate,
                return_tensors='pt',
                padding=True,
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.emo_encoder(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            pooled_embedding = torch.mean(last_hidden_state, dim=1)
            return pooled_embedding

        # speechbrain backend
        if waveform_tensor.ndim == 1:
            signal = waveform_tensor.unsqueeze(0)  # [1, T]
        else:
            # assume [C, T] after mono handling -> make batch
            signal = waveform_tensor[:1, ...]  # ensure [1, T]
        wav_lens = torch.tensor([1.0], device=self.device)

        # Ensure tensors live on the exact device as wav2vec2 weights
        try:
            target_device = next(self.sb_model.mods.wav2vec2.parameters()).device
        except Exception:
            target_device = self.device
        # Move full model and inputs to the same device
        self._ensure_sb_on_device(target_device)
        signal = signal.to(target_device)
        wav_lens = wav_lens.to(target_device)

        with torch.no_grad():
            embeddings = self.sb_model.encode_batch(signal, wav_lens=wav_lens)
        embeddings = embeddings.squeeze(0)
        return embeddings

    def _ensure_sb_on_device(self, device):
        """Best-effort move of SpeechBrain model and common submodules to device."""
        try:
            if hasattr(self, 'sb_model') and self.sb_model is not None:
                # Update device attribute if present
                if hasattr(self.sb_model, 'device'):
                    try:
                        self.sb_model.device = device
                    except Exception:
                        pass
                # Move top-level
                try:
                    self.sb_model.to(device)
                except Exception:
                    pass
                # Move known submodules
                mods = getattr(self.sb_model, 'mods', None)
                if mods is not None:
                    try:
                        mods.to(device)
                    except Exception:
                        pass
                    # Explicitly move wav2vec2 and nested HF model if present
                    w2v = getattr(mods, 'wav2vec2', None)
                    if w2v is not None:
                        try:
                            w2v.to(device)
                        except Exception:
                            pass
                        inner = getattr(w2v, 'model', None)
                        if inner is not None and hasattr(inner, 'to'):
                            try:
                                inner.to(device)
                            except Exception:
                                pass
                    for name in ('output_mlp', 'classifier'):
                        sub = getattr(mods, name, None)
                        if sub is not None and hasattr(sub, 'to'):
                            try:
                                sub.to(device)
                            except Exception:
                                pass
        except Exception:
            # Ignore; encode step will raise if something is still mismatched
            pass

    def forward(
        self,
        *,
        generated_wav=None,
        ref_emotion_embeddings=None,
        sample_rate=None,
        **kwargs,
    ):
        loss_dict = super().forward(**kwargs)

        if (
            self.use_emotion_consistency_loss
            and generated_wav is not None
            and ref_emotion_embeddings is not None
        ):
            loss_emo_con_sum = 0.0

            for i in range(generated_wav.size(0)):
                gen_wav_sample = generated_wav[i]
                ref_emo_emb_sample = (
                    ref_emotion_embeddings[i].to(self.device).squeeze()
                )

                gen_emo_emb = self._get_emotion_embedding_from_tensor(
                    gen_wav_sample, sample_rate
                ).squeeze()

                loss_emo_con_sum += -torch.nn.functional.cosine_similarity(
                    gen_emo_emb, ref_emo_emb_sample, dim=0
                )

            mean_loss_emo_con = loss_emo_con_sum / generated_wav.size(0)
            loss_dict['loss_emo_con'] = mean_loss_emo_con
            loss_dict['loss'] += (
                mean_loss_emo_con * self.emotion_consistency_loss_alpha
            )

        return loss_dict
