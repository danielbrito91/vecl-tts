def patch_state_dict(state_dict):
    """
    Patches a state_dict from an older Coqui TTS version to match the new format
    for weight-normalized layers.
    Maps '...weight_g' -> '...parametrizations.weight.original0'
    Maps '...weight_v' -> '...parametrizations.weight.original1'
    """
    print('    > Patching state dictionary for version compatibility...')
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
    print('    > Patching complete.')
    return new_state_dict
