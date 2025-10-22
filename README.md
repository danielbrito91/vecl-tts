
# VECL-TTS: Voice and Emotion Controllable TTS

A simplified implementation of VECL-TTS for Brazilian Portuguese.

## Quick Start

```bash
# Install dependencies
uv sync

# Download all required files
uv run task download-all

# Train VECL model
uv run task train-vecl

# Train YourTTS model
uv run task train-yourtts
```

## Project Structure

The project is organized into the following directories:

```plaintext
.
├── .env.example       # Environment variable template
├── pyproject.toml     # Project configuration and dependencies
├── configs/           # Hydra configuration files
├── docs/              # Project documentation
├── scripts/           # Automation scripts (download, train)
├── tests/             # Test suite
└── vecl/              # Core application source code
    ├── config.py      # Configuration loading
    ├── data/          # Data loading and preparation
    ├── embeddings/    # Speaker and emotion embedding
    ├── evaluation/    # Model evaluation scripts
    ├── preprocessing/ # Text and audio preprocessing
    └── training/      # Model training logic
```

## Available Tasks

Run `uv run task -l` to see all available tasks:

- **Testing**: `test`, `lint`, `format`
- **Downloads**: `download-all`, `download-dataset`, `download-models`
- **Training**: `train-vecl`, `train-yourtts`, `train-local`
- **Utilities**: `clean`, `setup`, `list-artifacts`

## Configuration

The system uses Hydra for configuration. Key files:

- `configs/config.yaml` - Main configuration
- `configs/model/` - Model-specific configs
- `configs/paths/` - Path configurations

Override any config value from command line:

```bash
uv run task train model=vecl training.batch_size=8 training.learning_rate=1e-4
```

## Storage Backends

The download system supports multiple backends:

```bash
# Local (default)
uv run task download --backend local

# Local mirror (copies from a mirror folder into project paths)
uv run task download --backend local --local-mirror /path/to/mirror

# List what's available
uv run task list-artifacts
```

### Local-first artifacts (no S3)

If you are not using S3, place your files at these paths so that `uv run task download-all` can validate (or copy from a local mirror) without errors. Paths below assume `configs/paths/local.yaml`.

Required

```plaintext
# Dataset metadata
data/processed_24k/metadata.csv

# Pretrained YourTTS checkpoint and config
models/checkpoints_yourtts_cml_tts_dataset/best_model.pth
models/checkpoints_yourtts_cml_tts_dataset/config.json
models/checkpoints_yourtts_cml_tts_dataset/language_ids.json
```

Optional (speeds up training if you already have them)

```plaintext
# Precomputed embeddings (if you want to use them)
data/processed_24k/speaker_embeddings_patch.pth
data/processed_24k/emotions.pth
```

Notes

- The exact target paths come from `configs/paths/*.yaml`:
  - `paths.dataset_path` (default local: `data/processed_24k`)
  - `paths.metadata_file` (default: `metadata.csv`)
  - `paths.pretrained_checkpoint_dir` (default local: `models`)
  - `paths.restore_path`, `paths.pretrained_config_path`, `paths.language_ids_file`
- If you prefer to copy from a local mirror, structure your mirror like this and pass `--local-mirror /path/to/mirror`:

```plaintext
/path/to/mirror/
  tts/cml-tts/processed_24k.tar.gz                           # optional dataset tar; if not present, just place metadata directly as above
  tts/cml-tts/checkpoints_yourtts_cml_tts_dataset.tar.bz      # model checkpoint archive (will extract into models/)
  tts/yourtts/embeddings/speaker_embeddings_patch.pth         # optional
  tts/vecl-tts/embeddings/emotion_embeddings.pth              # optional
```

Quick local-only run

```bash
# Ensure files exist at the required paths (see above)
uv run task download-all

# Then train
uv run task train-vecl
```

## Environment Variables

Create a `.env` file from the `.env.example` template to set your local environment variables.

- `S3_BUCKET_NAME`: S3 bucket for downloads/uploads.
- `WANDB_ENTITY`: Weights & Biases entity for experiment tracking.
- `OUTPUT_PATH`: Directory to save model outputs.
- `DATASET_PATH`: Path to the dataset.

## Development

```bash
# Run tests
uv run task test

# Format code
uv run task format

# Clean outputs
uv run task clean
```
