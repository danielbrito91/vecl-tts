#!/usr/bin/env python3
"""
Quick test script to verify the language manager fix.
"""

from pathlib import Path

from hydra import compose, initialize

# Import the refactored modules
from vecl.config import AppConfig
from vecl.models.config import VeclConfig
from vecl.models.vecl import Vecl


def test_language_manager():
    print('🧪 Testing language manager fix...')

    # Load configuration
    with initialize(config_path='configs', version_base=None):
        cfg = compose(config_name='local_test')
        config = AppConfig(**cfg)

    # Create VECL config with language management enabled
    vecl_config = VeclConfig()
    vecl_config.model_args.use_language_embedding = True
    vecl_config.model_args.language_ids_file = str(
        config.paths.language_ids_file
    )

    print(f'📁 Language IDs file: {vecl_config.model_args.language_ids_file}')
    print(
        f'📁 File exists: {Path(vecl_config.model_args.language_ids_file).exists()}'
    )

    # Test the init_from_config method
    try:
        model, updated_config = Vecl.init_from_config(
            config=vecl_config,
            samples=[],  # Empty samples for testing
        )

        if model.language_manager:
            lang_count = len(model.language_manager.name_to_id)
            languages = list(model.language_manager.name_to_id.keys())
            print('✅ Language manager successfully initialized!')
            print(f'   Languages ({lang_count}): {languages}')

            # Test specific languages
            for lang in ['en', 'pt-br']:
                if lang in model.language_manager.name_to_id:
                    lang_id = model.language_manager.name_to_id[lang]
                    print(f'   ✅ {lang} -> ID {lang_id}')
                else:
                    print(f'   ❌ {lang} not found')
        else:
            print('❌ Language manager is None')

    except Exception as e:
        print(f'❌ Error testing language manager: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    test_language_manager()
