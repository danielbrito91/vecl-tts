
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
# S3 (default)
uv run task download --backend s3

# Local mirror
uv run task download --backend local --local-mirror /path/to/mirror

# List what's available
uv run task list-artifacts
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
