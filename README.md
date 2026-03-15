# VECL-TTS: Voice and Emotion Controllable TTS for Brazilian Portuguese

Implementation of cross-lingual emotional text-to-speech synthesis for Brazilian Portuguese, adapting the [VECL-TTS](https://arxiv.org/abs/2407.18684) architecture with [YourTTS](https://arxiv.org/abs/2112.02418) as the backbone model.

This repository accompanies the paper accepted at **PROPOR 2026**.

## Pretrained Models

Pretrained model checkpoints will be available on [Hugging Face](https://huggingface.co/) shortly after publication.

| Model | Description | Link |
|-------|-------------|------|
| YourTTS (fine-tuned) | Fine-tuned on Portuguese emotional speech | *Coming soon* |
| YourTTS (fine-tuned + tokens) | Fine-tuned with emotional text tokens | *Coming soon* |
| VECL-TTS (α=1) | With emotion consistency loss (α=1) | *Coming soon* |
| VECL-TTS (α=9) | With emotion consistency loss (α=9) | *Coming soon* |

## Quick Start

```bash
# Install dependencies
uv sync

# Download required artifacts
uv run task download-all

# Train VECL model
uv run task train-vecl

# Train YourTTS model
uv run task train-yourtts
```

## Project Structure

```plaintext
.
├── configs/           # Hydra configuration files
├── scripts/           # Training and download scripts
├── tests/             # Test suite
└── vecl/              # Core source code
    ├── config.py      # Configuration schema
    ├── data/          # Data loading and preparation
    ├── embeddings/    # Speaker and emotion embedding extraction
    ├── evaluation/    # Evaluation utilities
    ├── models/        # VECL-TTS and YourTTS model definitions
    ├── preprocessing/ # Text and audio preprocessing
    └── training/      # Training loop and utilities
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
uv run task train model=vecl training.batch_size=8 training.learning_rate=1e-4
```

Key configuration files:

- `configs/config.yaml` — Main configuration
- `configs/model/` — Model-specific configs (`vecl.yaml`, `yourtts.yaml`)
- `configs/paths/` — Path configurations

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{brito2026vecl,
  title={Síntese de Voz Emocional Multi-Idioma para Português Brasileiro: Uma Análise Comparativa de Abordagens de Ajuste Fino},
  author={Brito, Daniel and Leal, Sidney and Candido Junior, Arnaldo},
  booktitle={Proceedings of PROPOR 2026},
  year={2026}
}
```
