"""
Embedding computation utilities for VECL-TTS.
"""

from .emotion import (
    EmotionEmbedding,
    compute_emotion_embeddings,
)
from .speaker import compute_speaker_embeddings

__all__ = [
    'compute_speaker_embeddings',
    'compute_emotion_embeddings',
    'EmotionEmbedding',
]
