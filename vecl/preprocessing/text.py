import re

import pandas as pd
from num2words import num2words


class TextPreprocessor:
    @staticmethod
    def preprocess(text: str, language: str) -> str:
        """Minimal preprocessing: lowercase + numbers to words"""
        if not isinstance(text, str):
            text = str(text)

        text = text.lower().strip().replace(')', '').replace('(', '')

        lang_map = {'pt-br': 'pt_BR', 'en': 'en'}
        num2words_lang = lang_map.get(language.lower())
        if num2words_lang is None:
            raise ValueError(
                f'Language {language} not supported for num2words'
            )

        def replace_number(match):
            try:
                return num2words(int(match.group()), lang=num2words_lang)
            except Exception as e:
                print(f'Error replacing number: {e}')
                return match.group()

        text = re.sub(r'\b\d+\b', replace_number, text)
        return text

    @staticmethod
    def preprocess_dataset(
        metadata: pd.DataFrame,
        text_col: str = 'transcription',
        lang_col: str = 'language',
    ) -> pd.DataFrame:
        """Apply preprocessing to a DataFrame."""
        metadata = metadata.copy()
        metadata['normalized_transcription'] = metadata.apply(
            lambda row: TextPreprocessor.preprocess(
                row[text_col], row[lang_col]
            ),
            axis=1,
        )
        return metadata
