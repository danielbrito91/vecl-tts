import os

import torch
import torchaudio
from torchaudio.functional import resample as ta_resample
from TTS.tts.models.vits import VitsDataset


def safe_load_audio(file_path: str, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    """Load audio and ensure waveform is in [-1, 1] with shape [1, T].

    This bypasses the strict assertion in Coqui-TTS' `load_audio` by
    normalizing/clamping potentially non-normalized PCM files.
    """
    waveform, sample_rate = torchaudio.load(file_path)  # [C, T], dtype may vary

    # Mixdown to mono
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to float
    waveform = waveform.float()

    # If it looks like raw int16 range, scale to [-1, 1]
    max_abs = waveform.abs().max()
    if max_abs > 1.0:
        # Heuristic: values >> 1 imply integer PCM scale
        if max_abs > 256:
            waveform = waveform / 32768.0
        else:
            waveform = waveform / max_abs

    # Final clamp to be safe
    waveform = torch.clamp(waveform, -1.0, 1.0)

    # Ensure [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Optional resample to target sample rate
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        # torchaudio resample expects [C, T]
        waveform = ta_resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )
        sample_rate = target_sample_rate

    return waveform, sample_rate


class VeclDataset(VitsDataset):
    """Custom dataset for VECL-TTS that loads pre-computed emotion embeddings."""

    def __init__(
        self, model_args, emotion_embedding_file=None, *args, **kwargs
    ):
        super().__init__(model_args=model_args, *args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.emotion_embeddings = None
        if emotion_embedding_file and os.path.exists(emotion_embedding_file):
            self.emotion_embeddings = torch.load(
                emotion_embedding_file, map_location='cpu'
            )

    def __getitem__(self, idx):
        item = self.samples[idx]
        target_sr = None
        if hasattr(self, 'config') and hasattr(self.config, 'audio'):
            target_sr = getattr(self.config.audio, 'sample_rate', None)
        wav, _ = safe_load_audio(item['audio_file'], target_sample_rate=target_sr)

        if (
            self.model_args.encoder_sample_rate
            and wav.size(1) % self.model_args.encoder_sample_rate != 0
        ):
            wav = wav[
                :, : -int(wav.size(1) % self.model_args.encoder_sample_rate)
            ]

        relative_path = os.path.relpath(item['audio_file'], item['root_path'])
        token_ids = self.get_token_ids(idx, item['text'])

        return {
            'raw_text': item['text'],
            'token_ids': token_ids,
            'token_len': len(token_ids),
            'wav': wav,
            'speaker_name': item['speaker_name'],
            'language_name': item['language'],
            'audio_unique_name': item['audio_unique_name'],
            'relative_path': relative_path,
        }

    def collate_fn(self, batch):
        """Custom collate_fn that pads the batch and adds emotion embeddings."""
        B = len(batch)
        batch_items = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch_items['wav']]),
            dim=0,
            descending=True,
        )

        max_text_len = max(len(x) for x in batch_items['token_ids'])
        token_lens = torch.LongTensor([
            len(x) for x in batch_items['token_ids']
        ])
        token_padded = torch.LongTensor(B, max_text_len).zero_() + self.pad_id
        token_rel_lens = token_lens / token_lens.max()

        max_wav_len = max(x.size(1) for x in batch_items['wav'])
        wav_padded = torch.FloatTensor(B, 1, max_wav_len).zero_()
        wav_lens = torch.LongTensor([x.size(1) for x in batch_items['wav']])
        wav_rel_lens = wav_lens / wav_lens.max()

        for i in range(B):
            idx = ids_sorted_decreasing[i]
            token = batch_items['token_ids'][idx]
            token_padded[i, : len(token)] = torch.LongTensor(token)
            wav = batch_items['wav'][idx]
            wav_padded[i, :, : wav.size(1)] = wav

        emotion_embeds = None
        if self.emotion_embeddings:
            emotion_embeds_list = []
            for i in range(B):
                idx = ids_sorted_decreasing[i]
                rel_path = batch_items['relative_path'][idx]
                if rel_path in self.emotion_embeddings:
                    emotion_embeds_list.append(
                        self.emotion_embeddings[rel_path]
                    )
                else:
                    emotion_embeds_list.append(
                        torch.zeros(self.model_args.emotion_embedding_dim)
                    )
            emotion_embeds = torch.stack(emotion_embeds_list)

        return {
            'tokens': token_padded,
            'token_lens': token_lens,
            'token_rel_lens': token_rel_lens,
            'waveform': wav_padded,
            'waveform_lens': wav_lens,
            'waveform_rel_lens': wav_rel_lens,
            'speaker_names': [
                batch_items['speaker_name'][i] for i in ids_sorted_decreasing
            ],
            'language_names': [
                batch_items['language_name'][i] for i in ids_sorted_decreasing
            ],
            'audio_unique_names': [
                batch_items['audio_unique_name'][i]
                for i in ids_sorted_decreasing
            ],
            'emotion_embeddings': emotion_embeds,
        }
