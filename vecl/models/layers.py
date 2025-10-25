"""
Neural network layers and components for VECL-TTS model architecture.
"""

from torch import nn


class EmotionProj(nn.Module):
    """
    A projection layer to map emotion embeddings to a target dimension.
    This is used in VECL-TTS to project raw emotion embeddings (e.g., 1024-dim)
    to the model's expected dimension (e.g., 512-dim).
    """

    def __init__(self, input_dim=768, output_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        print(f'Initialized projection layer: {input_dim} -> {output_dim}')

    def forward(self, x):
        """
        Projects the input tensor `x`.
        Args:
            x: Input tensor of shape (..., input_dim)
        Returns:
            Projected tensor of shape (..., output_dim)
        """
        # Project the embedding
        projected_x = self.proj(x)
        return projected_x
