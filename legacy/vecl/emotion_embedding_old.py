import os

import torch
import torchaudio
from torch import nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SER_MODEL_NAME = (
    'alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition'
)


class EmotionEmbedding:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model, self.feature_extractor = self.get_emotion_model()

    def get_emotion_model(self):
        # Load the feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            SER_MODEL_NAME
        )
        model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_NAME)

        model.to(self.device)

        return model, feature_extractor

    def get_emotion_embedding(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.feature_extractor.sampling_rate,
            )
            waveform = resampler(waveform)

        # Ensure the waveform is mono (single channel)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure the waveform is a 1D tensor for the feature extractor
        waveform = waveform.squeeze(0)

        # Extract features
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors='pt',
            padding=True,
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Perform inference asking for hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # The last element corresponds to the output of the final transformer layer.
        last_hidden_state = outputs.hidden_states[-1]

        # Aggregate the sequence embeddings into a single embedding
        # Perform mean pooling across the sequence dimension (dim=1)
        pooled_embedding = torch.mean(last_hidden_state, dim=1)

        return pooled_embedding


def emotion_consistency(gen_audio_path, ref_audio_path):
    if not os.path.exists(gen_audio_path):
        raise FileNotFoundError(f'File not found: {gen_audio_path}')
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f'File not found: {ref_audio_path}')

    em_embedding = EmotionEmbedding()
    gen_embedding = em_embedding.get_emotion_embedding(gen_audio_path)
    ref_embedding = em_embedding.get_emotion_embedding(ref_audio_path)

    return torch.nn.functional.cosine_similarity(gen_embedding, ref_embedding)


class EmotionProj(nn.Module):
    """
    A projection layer to map emotion embeddings to a target dimension.
    """

    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        print(f'Initialized projection layer: {input_dim} -> {output_dim}')

    def forward(self, x):
        """
        Projects the input tensor `x`.
        """
        # Project the embedding
        projected_x = self.proj(x)
        return projected_x
