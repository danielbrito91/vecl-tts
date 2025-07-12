# VECL-TTS: Voice Emotion and Cross-Lingual Transfer

This repository contains the implementation for VECL-TTS, a system for voice cloning with emotion and cross-lingual transfer capabilities.

## Prerequisites

On macOS, you may need to install LLVM for some dependencies to build correctly. You can do this with Homebrew:

```bash
brew install llvm
```

## How to Train

The main training script is `scripts/train_model.py`, which uses [Hydra](https://hydra.cc/) for configuration management. You can run different training configurations and override parameters from the command line.

### Running a Local Test

For a quick test run that uses minimal resources and doesn't log to external services like W&B or S3, use the `local_test` configuration. The command you used is correct:

```bash
uv run -m scripts.train_model --config-name local_test
```
You can also run it by pointing to the script file directly:
```bash
uv run python scripts/train_model.py --config-name local_test
```
This is ideal for debugging and verifying that the training loop runs without errors.

### Running the Default Training

To run the default training configuration (which uses the `vecl` model), simply execute the script:

```bash
uv run python scripts/train_model.py
```

### Overriding Configuration Parameters

You can override any parameter from the configuration files directly on the command line.

**Example 1: Switching to the `yourtts` model**
```bash
uv run python scripts/train_model.py model=yourtts
```

**Example 2: Changing training parameters**
You can modify any value from `configs/training/default.yaml` or other config files.
```bash
uv run python scripts/train_model.py training.learning_rate=1e-4 training.epochs=10

# Disable speaker embedding
uv run python scripts/train_model.py training.use_speaker_embedding=false
```

The output of each run will be saved in the directory specified by `paths.output_path`, inside a uniquely named folder based on the model and timestamp (e.g., `outputs/vecl_2023-10-27_19-00-00/`).

