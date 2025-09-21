#!/usr/bin/env python3
"""
Debug script to inspect speaker embeddings file format and find mismatched keys.
"""

from pathlib import Path

import torch


def inspect_embeddings(embeddings_file_path: str):
    """Inspect the structure of the speaker embeddings file."""
    print(f'🔍 Inspecting speaker embeddings file: {embeddings_file_path}')

    try:
        embeddings = torch.load(
            embeddings_file_path, map_location=torch.device('cpu')
        )
        print('✅ Successfully loaded embeddings file')
        print(f'📊 Total number of entries: {len(embeddings)}')

        # Show first 10 keys
        keys = list(embeddings.keys())
        print('\n🔑 First 10 keys:')
        for i, key in enumerate(keys[:10]):
            print(f'  {i + 1:2d}. {key}')

        # Look for the specific failing key
        failing_key = 'multilingual_custom_en#audio/0019_000223'
        print(f"\n🎯 Looking for failing key: '{failing_key}'")

        if failing_key in embeddings:
            print('✅ Found exact match!')
        else:
            print('❌ Exact match not found. Looking for similar keys...')

            # Find keys that contain parts of the failing key
            similar_keys = []
            filename_part = '0019_000223'
            dataset_part = 'multilingual_custom_en'

            for key in keys:
                if filename_part in key:
                    similar_keys.append(key)

            if similar_keys:
                print(
                    f"🔍 Found {len(similar_keys)} keys containing '{filename_part}':"
                )
                for key in similar_keys[:5]:  # Show first 5
                    print(f'  - {key}')
                if len(similar_keys) > 5:
                    print(f'  ... and {len(similar_keys) - 5} more')
            else:
                print(f"❌ No keys found containing '{filename_part}'")

            # Check what keys exist for this dataset
            dataset_keys = [k for k in keys if dataset_part in k]
            print(
                f"\n📋 Found {len(dataset_keys)} keys for '{dataset_part}' dataset"
            )
            if dataset_keys:
                print('Sample keys from this dataset:')
                for key in dataset_keys[:5]:
                    print(f'  - {key}')
                if len(dataset_keys) > 5:
                    print(f'  ... and {len(dataset_keys) - 5} more')

        # Show key patterns
        print('\n📝 Key patterns analysis:')
        patterns = {}
        for key in keys[:50]:  # Analyze first 50 keys
            if '#audio/' in key:
                dataset_part = key.split('#audio/')[0]
                patterns[dataset_part] = patterns.get(dataset_part, 0) + 1

        print('Dataset prefixes found:')
        for pattern, count in patterns.items():
            print(f'  - {pattern}: {count} files')

        # Check format of embeddings
        sample_key = keys[0]
        sample_embedding = embeddings[sample_key]
        print('\n📋 Sample embedding format:')
        print(f'  Key: {sample_key}')
        print(f'  Type: {type(sample_embedding)}')
        if isinstance(sample_embedding, dict):
            print(f'  Dict keys: {list(sample_embedding.keys())}')
            if 'embedding' in sample_embedding:
                print(
                    f'  Embedding shape: {sample_embedding["embedding"].shape}'
                )
        else:
            print(f'  Shape: {sample_embedding.shape}')

    except Exception as e:
        print(f'❌ Error loading embeddings file: {e}')


if __name__ == '__main__':
    # Look for speakers.pth in common locations
    possible_paths = [
        'speakers.pth',
        '/mnt/sagemaker-nvme/tts-dataset/speakers.pth',
        '/mnt/sagemaker-nvme/tts-checkpoints-multilingual/speakers.pth',
    ]

    for path in possible_paths:
        if Path(path).exists():
            inspect_embeddings(path)
            break
    else:
        print(
            '❌ Could not find speakers.pth file in any of the expected locations:'
        )
        for path in possible_paths:
            print(f'  - {path}')
