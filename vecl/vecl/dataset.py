import os
from pathlib import Path

import torch
from tqdm import tqdm
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import VitsDataset, load_audio

from vecl.vecl.emotion_embedding import EmotionEmbedding

# -----------------------------------------------------------------------------
# Remote storage settings (align with S3Trainer)
# -----------------------------------------------------------------------------
S3_BUCKET_NAME = 'hotmart-datascience-sagemaker'
# Default key where a pre-computed emotion embedding matrix might live.  Adjust
# in your own training environment if you upload it elsewhere.
S3_EMO_EMBED_KEY = 'tts/vecl-tts/emotion_embeddings.pth'


def compute_emotion_embeddings(
    dataset_configs_list, embeddings_file_path: Path
):
    """
    Computes emotion embeddings for each audio file in the dataset and saves
    them to a file.
    """
    # ----------------------------------------------------------------------
    # 1) If local file already present -> nothing to do
    # ----------------------------------------------------------------------
    if embeddings_file_path.exists():
        print(
            f'✅ Emotion embeddings file already exists at: {embeddings_file_path}'
        )
        return

    # ----------------------------------------------------------------------
    # 2) Attempt to download from S3 (optional) before expensive recompute
    # ----------------------------------------------------------------------
    try:
        import boto3

        print(
            f'>>> Attempting to fetch emotion embeddings from s3://{S3_BUCKET_NAME}/{S3_EMO_EMBED_KEY}'
        )
        s3_client = boto3.client('s3')
        # Check object exists
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=S3_EMO_EMBED_KEY)
        embeddings_file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            S3_BUCKET_NAME, S3_EMO_EMBED_KEY, str(embeddings_file_path)
        )
        print(f'✅ Downloaded emotion embeddings to: {embeddings_file_path}')
        return
    except Exception as e:
        print(
            f'⚠️  Could not download emotion embeddings from S3 ({e}). Will compute locally…'
        )

    print('>>> Computing emotion embeddings...')
    emotion_embedder = EmotionEmbedding()

    all_samples, _ = load_tts_samples(dataset_configs_list, eval_split=False)

    emotion_embeddings = {}
    for sample in tqdm(all_samples, desc='Computing Emotion Embeddings'):
        try:
            audio_file = sample['audio_file']
            root_path = sample['root_path']
            relative_path = os.path.relpath(audio_file, root_path)

            embedding = emotion_embedder.get_emotion_embedding(audio_file)
            emotion_embeddings[relative_path] = embedding.cpu()
        except Exception as e:
            print(f'\n    [ERROR] Failed to process {audio_file}: {e}')

    torch.save(emotion_embeddings, embeddings_file_path)
    print(
        f'\n✅ Embeddings for {len(emotion_embeddings)} files saved to: {embeddings_file_path}'
    )


def add_relative_path_to_samples(samples):
    """
    Adds 'relative_path' to each sample in a list of samples.
    This is used to retrieve emotion embeddings during training.
    """
    for sample in samples:
        sample['relative_path'] = os.path.relpath(
            sample['audio_file'], sample['root_path']
        )
    return samples


class VeclDataset(VitsDataset):
    """
    Custom dataset for VECL-TTS that loads pre-computed emotion embeddings.
    """

    def __init__(
        self, model_args, emotion_embedding_file=None, *args, **kwargs
    ):
        # CORRECTED: Explicitly pass model_args to the parent class constructor
        super().__init__(model_args=model_args, *args, **kwargs)

        # The parent constructor has now run and set up the tokenizer.
        # We can now safely access self.tokenizer.
        self.pad_id = self.tokenizer.characters.pad_id

        # Load emotion embeddings
        self.emotion_embeddings = None
        if emotion_embedding_file and os.path.exists(emotion_embedding_file):
            print(
                f'--> Loading emotion embeddings from: {emotion_embedding_file}'
            )
            self.emotion_embeddings = torch.load(
                emotion_embedding_file, map_location='cpu'
            )

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item['text']
        wav, _ = load_audio(item['audio_file'])

        # This check is from the original VitsDataset, it's good practice to keep it
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[
                    :,
                    : -int(wav.size(1) % self.model_args.encoder_sample_rate),
                ]

        # Get relative path for emotion embedding lookup
        # This assumes your 'root_path' is set correctly in the samples
        relative_path = os.path.relpath(item['audio_file'], item['root_path'])

        token_ids = self.get_token_ids(idx, item['text'])

        return {
            'raw_text': raw_text,
            'token_ids': token_ids,
            'token_len': len(token_ids),
            'wav': wav,
            'speaker_name': item['speaker_name'],
            'language_name': item['language'],
            'audio_unique_name': item['audio_unique_name'],
            'relative_path': relative_path,
        }

    def collate_fn(self, batch):
        """
        A custom collate_fn that pads the batch and adds emotion embeddings.
        This is based on your original implementation.
        """
        B = len(batch)
        batch_items = {k: [dic[k] for dic in batch] for k in batch[0]}

        # Sort by audio length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch_items['wav']]),
            dim=0,
            descending=True,
        )

        # Pad text sequences
        max_text_len = max([len(x) for x in batch_items['token_ids']])
        token_lens = torch.LongTensor([
            len(x) for x in batch_items['token_ids']
        ])
        token_rel_lens = token_lens / token_lens.max()
        token_padded = torch.LongTensor(B, max_text_len).zero_() + self.pad_id

        # Pad waveforms
        max_wav_len = max([x.size(1) for x in batch_items['wav']])
        wav_padded = torch.FloatTensor(B, 1, max_wav_len).zero_()
        wav_lens = torch.LongTensor([x.size(1) for x in batch_items['wav']])
        wav_rel_lens = wav_lens / wav_lens.max()

        # Populate padded tensors
        for i in range(B):
            idx = ids_sorted_decreasing[i]

            # Text
            token = batch_items['token_ids'][idx]
            token_padded[i, : len(token)] = torch.LongTensor(token)

            # Waveform
            wav = batch_items['wav'][idx]
            wav_padded[i, :, : wav.size(1)] = wav

        # Look up and add emotion embeddings
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
                    print(
                        f' [!] Warning: No emotion embedding found for {rel_path}. Using zeros.'
                    )
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
