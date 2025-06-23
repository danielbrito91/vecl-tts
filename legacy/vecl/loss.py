import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from TTS.config import Coqpit  # Make sure to import Coqpit
from TTS.tts.layers.losses import VitsGeneratorLoss

# Define the name of the SER model
SER_MODEL_NAME = (
    'alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition'
)


class VeclGeneratorLoss(VitsGeneratorLoss):
    """
    Generator loss for VECL-TTS, which efficiently integrates the emotion consistency logic.
    """

    def __init__(self, c: Coqpit):
        super().__init__(c)
        # Parameter lives under `model_args` in modern configs.
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

            # 1. Load the emotion model and feature extractor ONLY ONCE.
            self.emo_feat_ext = AutoFeatureExtractor.from_pretrained(
                SER_MODEL_NAME
            )
            self.emo_encoder = AutoModelForAudioClassification.from_pretrained(
                SER_MODEL_NAME
            )

            # Set to eval mode, freeze parameters, and move to the correct device.
            self.emo_encoder.eval()
            for param in self.emo_encoder.parameters():
                param.requires_grad = False
            self.emo_encoder.to(self.device)

    def _get_emotion_embedding_from_tensor(
        self, waveform_tensor: torch.Tensor, sample_rate: int
    ):
        """
        Computes emotion embedding from a waveform tensor in memory.
        This is the adapted version of your get_emotion_embedding method.
        """
        # Ensure waveform is on the correct device
        waveform_tensor = waveform_tensor.to(self.device)

        # Resample if necessary
        if sample_rate != self.emo_feat_ext.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.emo_feat_ext.sampling_rate,
            ).to(self.device)
            waveform_tensor = resampler(waveform_tensor)

        # Ensure mono and 1D for the feature extractor
        if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0)

        # Squeeze to get a 1D tensor
        waveform_tensor = waveform_tensor.squeeze()

        # Extract features
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
        # Compute all the standard VITS losses
        loss_dict = super().forward(**kwargs)

        # Compute and add the emotion consistency loss
        if (
            self.use_emotion_consistency_loss
            and generated_wav is not None
            and ref_emotion_embeddings is not None
        ):
            loss_emo_con_sum = 0.0

            # Process each item in the batch
            for i in range(generated_wav.size(0)):
                gen_wav_sample = generated_wav[i]
                # Collapse singleton dimensions so tensors are shaped [D]
                ref_emo_emb_sample = (
                    ref_emotion_embeddings[i].to(self.device).squeeze()
                )

                # Get embedding for the generated audio tensor
                gen_emo_emb = self._get_emotion_embedding_from_tensor(
                    gen_wav_sample, sample_rate
                ).squeeze()

                # Cosine similarity returns a 0-D tensor (scalar) when inputs are 1-D.
                loss_emo_con_sum += -torch.nn.functional.cosine_similarity(
                    gen_emo_emb, ref_emo_emb_sample, dim=0
                )

            # Average the loss over the batch
            mean_loss_emo_con = loss_emo_con_sum / generated_wav.size(0)
            loss_dict['loss_emo_con'] = mean_loss_emo_con
            loss_dict['loss'] += (
                mean_loss_emo_con * self.emotion_consistency_loss_alpha
            )

        return loss_dict
