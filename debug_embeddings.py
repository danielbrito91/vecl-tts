#!/usr/bin/env python3
"""
Debug script to examine speaker embeddings file and understand key structure.
"""

import sys
from pathlib import Path

import torch


def examine_embeddings_file(embeddings_path):
    """Examine the structure of a speaker embeddings file."""
    try:
        embeddings = torch.load(embeddings_path, map_location='cpu')

        print(f'📁 Embeddings file: {embeddings_path}')
        print(f'📊 Total embeddings: {len(embeddings)}')
        print(f'🔑 Type: {type(embeddings)}')

        # Show first 10 keys
        keys = list(embeddings.keys())
        print('\n🔍 First 10 keys:')
        for i, key in enumerate(keys[:10]):
            print(f'  {i + 1:2d}. {key}')

        # Show key patterns
        print('\n📈 Key patterns analysis:')
        extensions = set()
        prefixes = set()
        separators = set()

        for key in keys[:100]:  # Sample first 100 keys
            if '.' in key:
                ext = key.split('.')[-1]
                extensions.add(ext)

            if '/' in key:
                prefix = key.split('/')[0]
                prefixes.add(prefix)

            for char in ['#', '_', '-', '/']:
                if char in key:
                    separators.add(char)

        print(f'  Extensions found: {sorted(extensions)}')
        print(f'  Prefixes found: {sorted(prefixes)}')
        print(f'  Separators found: {sorted(separators)}')

        # Show embedding structure
        if keys:
            first_key = keys[0]
            first_embedding = embeddings[first_key]
            print('\n🎯 First embedding structure:')
            print(f'  Key: {first_key}')
            print(f'  Type: {type(first_embedding)}')

            if isinstance(first_embedding, dict):
                print(f'  Dict keys: {list(first_embedding.keys())}')
                if 'embedding' in first_embedding:
                    emb = first_embedding['embedding']
                    print(
                        f'  Embedding shape: {emb.shape if hasattr(emb, "shape") else "N/A"}'
                    )
                    print(f'  Embedding type: {type(emb)}')
            elif hasattr(first_embedding, 'shape'):
                print(f'  Shape: {first_embedding.shape}')
                print(f'  Type: {type(first_embedding)}')

        return True

    except Exception as e:
        print(f'❌ Error loading embeddings file: {e}')
        return False


if __name__ == '__main__':
    # Try different common paths
    potential_paths = [
        'data/processed_24k/speaker_embeddings_patch.pth',
        'data/processed_24k/speakers.pth',
        '${DATASET_PATH}/speakers.pth',
        '${DATASET_PATH}/speaker_embeddings_patch.pth',
    ]

    # If path provided as argument
    if len(sys.argv) > 1:
        potential_paths.insert(0, sys.argv[1])

    success = False
    for path_str in potential_paths:
        path = Path(path_str)
        if path.exists():
            print(f'🎯 Found embeddings file: {path}')
            success = examine_embeddings_file(path)
            break

    if not success:
        print('❌ No embeddings file found. Please provide path as argument:')
        print(
            'python debug_embeddings.py /path/to/your/speaker_embeddings.pth'
        )
