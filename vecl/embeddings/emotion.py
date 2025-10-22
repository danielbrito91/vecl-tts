import logging
import os
from abc import abstractmethod
from asyncio import AbstractChildWatcher
from pathlib import Path
from typing import Union

import torch
import torchaudio
from speechbrain.inference.interfaces import foreign_class
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

logger = logging.getLogger(__name__)


class BaseEmotionEmbedding(AbstractChildWatcher):
    @abstractmethod
    def get_emotion_embedding(self, audio_path) -> torch.Tensor:
        pass


class HFEmotionEmbedding(BaseEmotionEmbedding):
    def __init__(
        self,
        ser_model_name: str = 'alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition',
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model, self.feature_extractor = self.get_emotion_model(
            ser_model_name
        )

    def get_emotion_model(self, ser_model_name: str):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            ser_model_name
        )
        model = AutoModelForAudioClassification.from_pretrained(ser_model_name)

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


class SpeechBrainEmotionEmbedding(BaseEmotionEmbedding):
    def __init__(
        self,
        ser_model_name: str = 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
    ):
        self.model = foreign_class(
            source=ser_model_name,
            pymodule_file='custom_interface.py',
            classname='CustomEncoderWav2vec2Classifier',
        )
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

    def get_emotion_embedding(self, audio_path) -> torch.Tensor:
        signal = self.model.load_audio(audio_path)

        signal = signal.unsqueeze(0)
        signal = signal.to(self.model.device)

        wav_lens = torch.tensor([1.0], device=self.model.device)

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode_batch(signal, wav_lens=wav_lens)

        embeddings = embeddings.squeeze(0)
        return embeddings


def load_emotion_embedder(
    ser_model_name: str,
) -> Union[SpeechBrainEmotionEmbedding, HFEmotionEmbedding]:
    if ser_model_name.startswith('speechbrain'):
        return SpeechBrainEmotionEmbedding(ser_model_name=ser_model_name)
    elif ser_model_name.startswith('alefiury'):
        return HFEmotionEmbedding(ser_model_name=ser_model_name)
    else:
        raise ValueError(f'Invalid SER model name: {ser_model_name}')


def compute_emotion_embeddings(
    dataset_configs: list[BaseDatasetConfig],
    embeddings_file_path: Path,
    emotion_embedder: BaseEmotionEmbedding,
):
    all_samples, _ = load_tts_samples(dataset_configs, eval_split=False)
    emotion_embeddings = {}
    for sample in tqdm(all_samples, desc='Computing Emotion Embeddings'):
        try:
            audio_file = sample['audio_file']
            relative_path = os.path.relpath(audio_file, sample['root_path'])
            embedding = emotion_embedder.get_emotion_embedding(audio_file)
            emotion_embeddings[relative_path] = embedding.cpu()
        except Exception as e:
            logger.error(f'Failed to process {audio_file}: {e}')

    torch.save(emotion_embeddings, embeddings_file_path)
    logger.info(f'Emotion embeddings saved to: {embeddings_file_path}')
