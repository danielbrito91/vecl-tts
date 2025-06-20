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

### ▶️ Step 2.1: Unify Data Processing Utilities

-   **Action:**
    1.  Create a new file: `vecl/data_utils.py`.
    2.  Merge the content of the following files into `vecl/data_utils.py`:
        -   `vecl/dataset/preprocess_text.py` (`TextPreprocessor`)
        -   `vecl/dataset/preprocess_audio.py` (`AudioPreprocessor`)
        -   `vecl/yourtts/dataset.py` (`prepare_dataset_configs`, `compute_speaker_embeddings`)
        -   `vecl/vecl/dataset.py` (`compute_emotion_embeddings`, `VeclDataset`)
    3.  Resolve all imports and helper function calls within the new unified file.
-   **Verification Test:**
    -   **Unit Tests:** Create a `tests/test_data_utils.py` file. Write a small unit test for `TextPreprocessor` with a sample sentence and for `AudioPreprocessor` with a short, silent WAV file.
    -   **Integration Test:** Create a dummy metadata CSV file with 2-3 entries. Run the `prepare_dataset_configs` function and assert that the output is a list of `BaseDatasetConfig` objects with the expected structure.

### Step 2.2: Unify Model & Checkpoint Handling

-   **Action:**
    1.  Create a new file: `vecl/model_utils.py`.
    2.  Move the `Vecl` model definition from `vecl/vecl/vecl.py` into this new file.
    3.  Move the `EmotionProj` and `EmotionEmbedding` classes from `vecl/vecl/emotion_embedding.py` into this file.
    4.  Move helper functions like `patch_state_dict` and S3 download logic from the training scripts into this file.
    5.  Create a unified `load_model_and_config` function that can handle loading both VECL and YourTTS models.
-   **Verification Test:**
    -   **Integration Test:** In `tests/test_model_utils.py`, write a test that uses `load_model_and_config` to load a real VECL checkpoint. The test passes if the function returns a valid model object and configuration without errors. Check that `model.has_emotion_proj` is true. Run it again with a YourTTS checkpoint and verify the opposite.

### Step 2.3: Unify the Trainer

-   **Action:**
    1.  Create a new file: `vecl/trainer.py`.
    2.  Move the `S3Trainer` class from `vecl/yourtts/s3_trainer.py` into this file.
    3.  Merge the main training loops and logic from `train_vecl.py` and `train_yourtts_custom.py` into a single, new `UnifiedTrainer` class. This class will take the `AppConfig` object and handle model and loss function selection based on `config.model.type`.
-   **Verification Test:**
    -   **Integration Test:** Create a `scripts/02_train_model.py` entry point. Run a training job for a **single step** on a tiny (2-sample) dataset.
        1.  Run with the `vecl_train` config. It passes if it completes one step and computes a valid loss.
        2.  Run with the `yourtts_finetune` config. It passes if it also completes one step.

---

## Phase 3: Refactor Entry Points

**Goal:** Create clean, configuration-driven scripts for each major workflow.

### Step 3.1: Create Unified Preprocessing Script

-   **Action:** Create `scripts/01_preprocess_data.py`. This script should load the configuration and call the necessary functions from `vecl/data_utils.py` to prepare a dataset.
-   **Verification Test:** Run the script on a small, raw dataset (e.g., a few WAV files and a simple CSV). The test passes if the script correctly generates the processed audio files and the final `metadata.csv` in the output directory.

### Step 3.2: Create Unified Inference Script

-   **Action:**
    1.  Refactor the `generate` and `main` functions from `infer_vecl.py` and `yourtts/infer.py` into a single `scripts/03_run_inference.py`.
    2.  This script will use the unified `synthesize` function from `vecl/model_utils.py`.
-   **Verification Test:** Run `scripts/03_run_inference.py` with a sample sentence and a reference audio. The test passes if it generates a valid, non-empty `.wav` file.

### Step 3.3: Create Unified Evaluation Script

-   **Action:**
    1.  Create `vecl/evaluation.py` and move the `emotion_consistency` logic and speaker similarity calculation from the notebooks into it.
    2.  Create `scripts/04_run_evaluation.py`. This script will load a test set from a metadata file, use the inference script to generate audio for each entry, and then use `vecl/evaluation.py` to compute all metrics.
    3.  The script will save the results to a single, clean `results.csv` file.
-   **Verification Test:** Run the script on a small test set (2-3 audio files). The test passes if it generates the output audio and a `results.csv` file with the correct headers and calculated scores.

---

## Phase 4: Final Cleanup

**Goal:** Remove all old and redundant files and directories.

### Step 4.1: Delete Redundant Code

-   **Action:** After all tests for the above phases are passing, delete the following files and directories:
    -   `vecl/vecl/`
    -   `vecl/yourtts/`
    -   `vecl/dataset/`
    -   `vecl/train_vecl.py`
    -   `vecl/infer_vecl.py`
    -   `notebooks/15_eval_vecl_5000.ipynb`
    -   `notebooks/07-eval-cml.ipynb`
-   **Verification Test:** Re-run all tests created in the previous phases. If everything still passes, the refactoring is complete and successful. 