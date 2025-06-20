import torch

# Define paths to the checkpoints to compare
# These should match the paths in your infer.py script
CML_CHECKPOINT_PATH = (
    'models/checkpoints_yourtts_cml_tts_dataset/best_model.pth'
)
FINETUNE_CHECKPOINT_PATH = 'models/finetune/checkpoint_10_02.pth'


def patch_state_dict(state_dict):
    """
    Patches a state_dict from an older Coqui TTS version to match the new format
    for weight-normalized layers.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'weight_g' in k:
            new_k = k.replace(
                '.weight_g', '.parametrizations.weight.original0'
            )
        elif 'weight_v' in k:
            new_k = k.replace(
                '.weight_v', '.parametrizations.weight.original1'
            )
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def compare_checkpoints(path1, path2):
    """
    Loads two checkpoints, patches their state dicts, and compares their contents.
    """
    print('--- Comparing Checkpoints ---')
    print(f'  > Pre-trained: {path1}')
    print(f'  > Fine-tuned:  {path2}\n')

    # Load checkpoints
    try:
        cp1 = torch.load(path1, map_location='cpu')
        cp2 = torch.load(path2, map_location='cpu')
    except FileNotFoundError as e:
        print(
            '❌ ERROR: Could not find a checkpoint file. Please check the paths.'
        )
        print(f'  > {e}')
        return

    # Get model state dicts
    state1 = cp1.get('model', cp1)
    state2 = cp2.get('model', cp2)

    # Patch for version compatibility
    print('> Patching state dicts for a fair comparison...')
    state1 = patch_state_dict(state1)
    state2 = patch_state_dict(state2)
    print('> Patching complete.\n')

    keys1 = set(state1.keys())
    keys2 = set(state2.keys())

    # --- Key Comparison ---
    print('--- 1. Key Comparison ---')
    if keys1 == keys2:
        print(
            '✅ The set of keys (layer names) is identical in both checkpoints.'
        )
    else:
        print('⚠️ The keys (layer names) are different!')
        missing_in_ft = keys1 - keys2
        added_in_ft = keys2 - keys1
        if missing_in_ft:
            print('\n  > Layers in pre-trained but MISSING in fine-tuned:')
            for key in sorted(list(missing_in_ft)):
                print(f'    - {key}')
        if added_in_ft:
            print('\n  > Layers ADDED in fine-tuned but not in pre-trained:')
            for key in sorted(list(added_in_ft)):
                print(f'    - {key}')
    print('')

    # --- Shape and Value Comparison ---
    print('--- 2. Shape and Value Comparison (for common layers) ---')
    common_keys = keys1.intersection(keys2)
    shape_mismatches = 0
    value_differences = 0

    for key in sorted(list(common_keys)):
        t1 = state1[key]
        t2 = state2[key]

        if t1.shape != t2.shape:
            print(f'  > ⚠️ Shape Mismatch for key `{key}`:')
            print(f'    - Pre-trained: {t1.shape}')
            print(f'    - Fine-tuned:  {t2.shape}')
            shape_mismatches += 1
        elif not torch.equal(t1, t2):
            value_differences += 1

    if shape_mismatches == 0:
        print('✅ All common layers have matching shapes.')
    else:
        print(f'❌ Found {shape_mismatches} layers with shape mismatches.')

    if value_differences == 0:
        print(
            '⚠️ All common layers have identical weights. The model was not updated during fine-tuning.'
        )
    else:
        print(
            f'✅ Found {value_differences} layers with updated weights, confirming that fine-tuning occurred.'
        )

    print('\n--- Exploration Complete ---')


if __name__ == '__main__':
    compare_checkpoints(CML_CHECKPOINT_PATH, FINETUNE_CHECKPOINT_PATH)
