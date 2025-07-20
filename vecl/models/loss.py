import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from TTS.config import Coqpit
from TTS.tts.layers.losses import VitsGeneratorLoss

SER_MODEL_NAME = (
    'alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition'
)


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

            self.emo_feat_ext = AutoFeatureExtractor.from_pretrained(
                SER_MODEL_NAME
            )
            self.emo_encoder = AutoModelForAudioClassification.from_pretrained(
                SER_MODEL_NAME
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
        """
        waveform_tensor = waveform_tensor.to(self.device)

        if sample_rate != self.emo_feat_ext.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.emo_feat_ext.sampling_rate,
            ).to(self.device)
            waveform_tensor = resampler(waveform_tensor)

        if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0)

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
