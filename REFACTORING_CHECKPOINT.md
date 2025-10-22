# Refactoring Plan: `vecl/model/checkpoint.py`

This document outlines a minimal, step-by-step refactoring plan for `vecl/model/checkpoint.py` and its related strategies. The goal is to improve generality and maintainability by making small, incremental changes that can be tested at each stage.

---

### Step 1: Unify Speaker Encoder File Resolution Logic

**Goal:** Make a small, safe improvement to the existing file resolution helper to reduce code duplication.

**Problem:** The current `_resolve_speaker_encoder_paths` function has separate `if` blocks for handling `model_se.pth` and `config_se.json`, which is slightly repetitive.

**Action:**
1.  Modify the `_resolve_speaker_encoder_paths` function in `vecl/model/checkpoint.py`.
2.  Use a loop to iterate through a list containing the model and config file information. This will unify the logic for checking, moving, or downloading each file.

**Verification:**
-   Run the existing test suite (e.g., `tests/model/test_checkpoint.py`). All tests related to loading models should pass without any changes, as this is a purely internal refactor with no change in functionality.

---

### Step 2: Introduce a Generic, Backward-Compatible Artifact Resolver

**Goal:** Generalize the file **resolution** logic to handle any type of dependency (not just speaker encoders) while ensuring old checkpoints still work.

**Problem:** The `_resolve_speaker_encoder_paths` function is hardcoded. A future model might need different artifacts, which would require changing this function again.

**Action:**
1.  In `vecl/model/checkpoint.py`, create a new function `_resolve_artifacts(config: AppConfig, model_config: DictConfig, checkpoint_dir: Path) -> Dict[str, Path]`.
2.  **Inside this new function, add a fallback mechanism:**
    ```python
    if not hasattr(model_config, 'artifacts'):
        # Fallback for old checkpoints
        se_model, se_config = _resolve_speaker_encoder_paths(config) # The original function
        return {
            'speaker_encoder_model_path': se_model,
            'speaker_encoder_config_path': se_config,
        }
    # ... new generic logic continues here ...
    ```
3.  Implement the generic logic to read from a new `artifacts` dictionary in the model's `config.json`. It will loop through the declared artifacts and resolve their paths.
4.  In `load_model_for_training` and `load_model_for_inference`, replace the call to `_resolve_speaker_encoder_paths` with a call to the new `_resolve_artifacts`. Then, loop through the returned dictionary to update the paths in the `model_config`.

**Verification:**
1.  **Backward Compatibility:** Run the entire existing test suite. All tests should pass because the fallback mechanism will be triggered for all current test cases.
2.  **New Functionality:** Write a new test case in `tests/model/test_checkpoint.py` specifically for `_resolve_artifacts`. This test should:
    -   Create a dummy `config.json` that *does* include an `artifacts` manifest (e.g., declaring a speaker encoder and a vocoder file).
    -   Assert that the function correctly resolves the paths for these declared artifacts.

---

### Step 3: Introduce Explicit, Backward-Compatible Model Typing

**Goal:** Move from implicit model type detection (`hasattr(...)`) to an explicit one, without breaking old checkpoints.

**Problem:** The line `is_vecl_model = hasattr(cfg.model_args, 'emotion_embedding_dim')` is brittle. It relies on a side-effect of the model's configuration.

**Action:**
1.  In `load_model_for_inference`, modify the model type detection logic.
2.  Prioritize checking for an explicit `model_type` field in the `config.json`, but keep the `hasattr` check as a fallback for older models.
    ```python
    # New logic
    model_type = cfg.get('model_type', None)
    if model_type == 'vecl':
        is_vecl_model = True
    elif model_type == 'yourtts':
        is_vecl_model = False
    else:
        # Fallback for older checkpoints without the 'model_type' field
        is_vecl_model = hasattr(cfg.model_args, 'emotion_embedding_dim')
    ```

**Verification:**
1.  **Backward Compatibility:** All existing tests for inference should pass, as they will use the fallback logic.
2.  **New Functionality:** Add a new test case for `load_model_for_inference` where the `config.json` mock contains `model_type: 'vecl'`. Assert that the `Vecl` model class is chosen for initialization.

---

### Step 4: Refactor State Dict Patching into a Dedicated Helper

**Goal:** Improve code organization by consolidating state dictionary modifications into a single, focused function.

**Problem:** The logic for patching the model's state dictionary (e.g., removing keys) is currently inside `load_model_for_training`.

**Action:**
1.  Create a new private helper function in `vecl/model/checkpoint.py` called `_patch_state_dict_for_training(state_dict, training_config)`.
2.  Move the following logic from `load_model_for_training` into this new function:
    -   The `if` condition that deletes `emb_l.weight` if `use_pretrained_lang_embeddings` is false.
3.  Modify `load_model_for_training` to call this new helper function.

**Note:** The logic for dynamically creating the `emotion_proj` layer should remain in `load_model_for_inference`, as it modifies the *model object* itself, not just the state dictionary before loading.

**Verification:**
-   Run all existing tests for `load_model_for_training`. They should all pass, as this is purely a code organization refactor with no functional changes.

---

By following these small, incremental steps, the model loading system can be significantly improved without introducing breaking changes, making the process safe and verifiable. 