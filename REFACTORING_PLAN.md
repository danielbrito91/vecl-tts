# VECL-TTS Refactoring Plan

This document outlines the step-by-step plan to refactor the `vecl-pt` repository. The goal is to create a modular, maintainable, and reproducible codebase by decoupling configuration, unifying duplicated logic, and creating clear, single-purpose entry points for each workflow.

---

## Phase 1: Configuration & Foundation

**Goal:** Decouple all hardcoded settings from the application code into a structured, secure, and validated configuration system.

### ✅ Step 1.1: Create New Directory Structure

-   **Action:** Create the primary directories for the new structure.
    ```bash
    mkdir -p configs/experiment configs/paths configs/params scripts
    ```
-   **Verification Test:** Run `ls -R` to confirm that the `configs` and `scripts` directories, along with their subdirectories, have been created at the project root.

### ✅ Step 1.2: Define Configuration Schema with Pydantic

-   **Action:** Create a new file, `vecl/config.py`, to define the entire configuration structure using Pydantic models. This provides type safety, validation, and auto-completion.
-   **Verification Test:** Create a temporary test script (`scripts/test_config_schema.py`) that imports the main `AppConfig` model from `vecl/config.py` and attempts to instantiate it with dummy data. The test passes if the object is created without errors.

### ✅ Step 1.3: Externalize Configuration to YAML

-   **Action:** Create a `configs/main.yaml` file to hold all default settings. Create experiment-specific files like `configs/experiment/vecl_train.yaml` and `configs/experiment/yourtts_finetune.yaml` that can override the defaults.
-   **Verification Test:** Write a script that uses a library like `PyYAML` to load `main.yaml` and an experiment YAML, merges them, and then uses the resulting dictionary to instantiate the `AppConfig` Pydantic model. The test passes if the configuration is loaded and parsed without validation errors.

### ✅ Step 1.4: Use Environment Variables for Secrets

-   **Action:**
    1.  Add `pydantic-settings` to the project dependencies.
    2.  Create a `.env.example` file to template the required environment variables.
    3.  Modify `vecl/config.py` to use `pydantic_settings.BaseSettings` to read critical paths (`OUTPUT_PATH`, `DATASET_PATH`) and secrets (`S3_BUCKET_NAME`, `WANDB_ENTITY`) from a `.env` file, making them mandatory.
    4.  Remove the hardcoded values from `configs/main.yaml`.
-   **Verification Test:**
    1.  Temporarily rename your `.env` file to `.env.bak`.
    2.  Run a script that loads the configuration. It **must fail** with a Pydantic `ValidationError` indicating that the required environment variables are missing.
    3.  Rename `.env.bak` back to `.env`.
    4.  Run the script again. It **must succeed** and correctly load the values from the file.

---

## Phase 2: Unify Core Logic

**Goal:** Consolidate duplicated code into a single, reusable `vecl` library.

### ✅ Step 2.1: Modularize Data Processing Utilities

-   **Action:**
    1.  **Break down the monolithic `data_utils.py`** into focused, modular components:
        -   `vecl/preprocessing/` - Text and audio preprocessing utilities (`TextPreprocessor`, `AudioPreprocessor`)
        -   `vecl/embeddings/` - Speaker and emotion embedding computation (`SpeakerEmbedding`, `EmotionEmbedding`)
        -   `vecl/dataset/` - Dataset preparation and custom dataset classes (`prepare_dataset_configs`, `VeclDataset`)
        -   `vecl/model/` - Neural network layers (`EmotionProj`)
        -   `vecl/evaluation/` - Evaluation metrics (`emotion_consistency`)
    2.  **Maintain backward compatibility** through a facade pattern in `vecl/data_utils.py` that imports and re-exports all public APIs.
    3.  **Improve separation of concerns** by having core functions accept specific parameters rather than entire configuration objects.
-   **Verification Test:**
    -   **Unit Tests:** Created comprehensive test suites:
        -   `tests/preprocessing/test_text.py` and `tests/preprocessing/test_audio.py`
        -   `tests/embeddings/test_speaker.py` and `tests/embeddings/test_emotion.py` 
        -   `tests/dataset/test_dataset_preparation.py` and `tests/dataset/test_vecl_dataset.py`
        -   `tests/evaluation/test_metrics.py`
        -   `tests/model/layers.py`
    -   **Integration Test:** All tests pass with shared fixtures managed through `tests/conftest.py`.
    -   **Enhancements:** Further refactored `vecl/embeddings/speaker.py` and `vecl/embeddings/emotion.py` to remove hardcoded model names and downloaders. Both modules now use the centralized `AppConfig` and `vecl/utils/downloader.py` to fetch assets from S3, improving consistency and removing redundant code.

### ✅ Step 2.2: Modularize Model & Checkpoint Handling

-   **Action:**
    1.  **Consolidated model components** into the `vecl/model/` directory:
        -   Moved the `Vecl` model definition from `vecl/vecl/vecl.py` to `vecl/model/vecl.py`.
        -   Moved related classes (`VeclConfig`, `VeclArgs`) to `vecl/model/config.py`.
        -   Moved `EmotionProj` to `vecl/model/layers.py`.
        -   Moved `VeclGeneratorLoss` to `vecl/model/loss.py`.
    2.  **Created a unified checkpoint loader** in `vecl/model/checkpoint.py`:
        -   The new `load_model_for_inference` function accepts the main `AppConfig` and replaces duplicated logic from `infer_vecl.py` and `yourtts/infer.py`.
        -   It dynamically loads either a `Vecl` or standard `YourTTS` model based on `config.model.type`.
        -   It robustly handles path resolution, state dict patching, and dynamic creation of missing layers (e.g., `emotion_proj`).
    3.  **Centralized asset downloading**:
        -   Created `vecl/utils/downloader.py` with a config-driven `download_s3_file` utility.
        -   Integrated this downloader into `vecl/model/checkpoint.py` to automatically fetch the speaker encoder if it's not found locally.
    4.  **Maintained the facade pattern** by updating `vecl/data_utils.py` to re-export all new model components and utilities, ensuring backward compatibility.
-   **Verification Test:**
    -   **Unit Tests:** Created `tests/embeddings/test_speaker.py` and `tests/embeddings/test_emotion.py` to validate that the refactored embedding modules correctly handle S3 downloads, configuration changes, and backwards compatibility with older embedding formats.
    -   **Integration Test:** The `load_model_for_inference` function was confirmed to correctly load both VECL and YourTTS models based on the `AppConfig`, demonstrating that the modular, configuration-driven approach is successful.

### Step 2.3: Modularize Training Components

-   **Action:**
    1.  **Create modular training structure** within the existing architecture:
        -   `vecl/training/` - Training-related components
            -   Move and refactor `S3Trainer` class from `vecl/yourtts/s3_trainer.py`
            -   Create a `UnifiedTrainer` class that leverages the modular components
            -   Extract common training utilities and loss functions
        -   **Leverage existing modules**: Use `vecl/model/`, `vecl/dataset/`, and `vecl/embeddings/` components
    2.  **Update the facade pattern** in `vecl/data_utils.py` to include training utilities.
    3.  **Configuration-driven training** where the trainer selects appropriate models and loss functions based on `config.model.type`.
-   **Verification Test:**
    -   **Unit Tests:** Create `tests/training/` with tests for trainer components and loss functions.
    -   **Integration Test:** Create a `scripts/02_train_model.py` entry point that runs single-step training on a tiny dataset for both VECL and YourTTS configurations.

---

## Phase 3: Refactor Entry Points

**Goal:** Create clean, configuration-driven scripts for each major workflow.

### Step 3.1: Create Unified Preprocessing Script

-   **Action:** Create `scripts/01_preprocess_data.py`. This script should:
    1.  Load the configuration using the established config system
    2.  Use the modular components from `vecl/preprocessing/`, `vecl/dataset/`, and `vecl/embeddings/` 
    3.  Maintain the facade pattern by importing through `vecl/data_utils.py` for backward compatibility
-   **Verification Test:** Run the script on a small, raw dataset (e.g., a few WAV files and a simple CSV). The test passes if the script correctly generates the processed audio files and the final `metadata.csv` in the output directory.

### Step 3.2: Create Unified Inference Script

-   **Action:**
    1.  Create `scripts/03_run_inference.py` that consolidates inference logic from `infer_vecl.py` and `yourtts/infer.py`.
    2.  **Leverage modular components**: Use model loading from `vecl/model/`, embeddings from `vecl/embeddings/`, and preprocessing from `vecl/preprocessing/`.
    3.  **Maintain facade compatibility** by importing through `vecl/data_utils.py`.
    4.  **Configuration-driven inference** that handles both VECL and YourTTS models based on config.
-   **Verification Test:** Run `scripts/03_run_inference.py` with a sample sentence and a reference audio. The test passes if it generates a valid, non-empty `.wav` file.

### Step 3.3: Create Unified Evaluation Script

-   **Action:**
    1.  **Expand the existing `vecl/evaluation/` module** (already created in Step 2.1) with additional metrics from notebooks.
    2.  Create `scripts/04_run_evaluation.py` that:
        -   Uses the modular evaluation components from `vecl/evaluation/`
        -   Leverages inference capabilities through the modular architecture
        -   Integrates with the existing embeddings and preprocessing modules
    3.  **Maintain facade compatibility** by ensuring evaluation functions are accessible through `vecl/data_utils.py`.
    4.  The script will save results to a clean `results.csv` file.
-   **Verification Test:** Run the script on a small test set (2-3 audio files). The test passes if it generates the output audio and a `results.csv` file with the correct headers and calculated scores.

---

## Phase 4: Final Cleanup

**Goal:** Remove all old and redundant files and directories.

### Step 4.1: Archive Redundant Code

-   **Action:** After all tests for the above phases are passing, move the following legacy directories to a new `legacy/` directory to preserve the original code while removing it from the active codebase:
    -   `legacy/vecl/` (originally `vecl/vecl/`)
    -   `legacy/yourtts/` (originally `vecl/yourtts/`)
    -   **Note:** All other redundant files from the original implementation (e.g., `infer_vecl.py`) will be deleted once they are fully replaced by the new scripted entry points.
-   **Verification Test:** Re-run all tests created in the previous phases. If everything still passes, the refactoring is complete and successful. 