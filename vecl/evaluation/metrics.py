"""
Evaluation metrics for VECL-TTS model performance assessment.
"""

import os

import torch

from vecl.embeddings.emotion import EmotionEmbedding


def emotion_consistency(
    gen_audio_path: str, ref_audio_path: str
) -> torch.Tensor:
    """
    Calculate emotion consistency between generated and reference audio.
    This metric computes the cosine similarity between emotion embeddings
    of the generated audio and reference audio. Higher values indicate
    better emotion preservation.
    Args:
        gen_audio_path: Path to the generated audio file
        ref_audio_path: Path to the reference audio file
    Returns:
        Cosine similarity score between emotion embeddings
    Raises:
        FileNotFoundError: If either audio file doesn't exist
    """
    if not os.path.exists(gen_audio_path):
        raise FileNotFoundError(
            f'Generated audio file not found: {gen_audio_path}'
        )
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(
            f'Reference audio file not found: {ref_audio_path}'
        )

    em_embedding = EmotionEmbedding()
    gen_embedding = em_embedding.get_emotion_embedding(gen_audio_path)
    ref_embedding = em_embedding.get_emotion_embedding(ref_audio_path)

    return torch.nn.functional.cosine_similarity(gen_embedding, ref_embedding)
