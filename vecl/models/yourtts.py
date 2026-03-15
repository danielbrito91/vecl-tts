from pathlib import Path
from typing import Dict, List, Union

from TTS.tts.models.vits import Vits


class YourTTS(Vits):
    def format_batch(self, batch: dict) -> dict:
        """
        Since we are using a custom dataset management, the sample's unique name
        is composed of the dataset_name and the audio_unique_name to avoid
        collisions. This method formats the batch to use this composite key
        when fetching d-vectors.

        We also need to ensure that audio_unique_name does not contain any
        directory prefix, aligning with how the speaker embedding keys are generated.
        """
        if 'audio_unique_names' in batch and 'dataset_names' in batch:
            composite_keys = [
                f'{d}#{Path(a).name}'
                for d, a in zip(
                    batch['dataset_names'], batch['audio_unique_names']
                )
            ]
            batch['audio_unique_names'] = composite_keys

        return super().format_batch(batch)

    def get_aux_input_from_test_sentences(self, sentence_info):
        """
        Override to handle speaker names with trailing whitespace/newlines
        and ensure text is a proper string.
        """
        import torch
        
        # Unpack sentence info
        text, utt_id, speaker_name, language = sentence_info
        
        # Ensure text is a string (not a tuple or other type)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Strip whitespace from speaker name (fixes '9351\n' issue)
        if isinstance(speaker_name, str):
            speaker_name = speaker_name.strip()
        
        # Create patched sentence info
        patched_info = (text, utt_id, speaker_name, language)
        
        # Call parent implementation with cleaned data
        aux_inputs = super().get_aux_input_from_test_sentences(patched_info)
        
        # Double-check that aux_inputs['text'] is a string
        if 'text' in aux_inputs and not isinstance(aux_inputs['text'], str):
            aux_inputs['text'] = str(aux_inputs['text']) if aux_inputs['text'] is not None else ""
        
        # Ensure language_id is properly set if language_manager exists
        # Fix for channel mismatch: model expects language embeddings
        if self.language_manager is not None and language:
            if language in self.language_manager.name_to_id:
                lang_id = self.language_manager.name_to_id[language]
                aux_inputs['language_id'] = torch.LongTensor([lang_id])
        
        return aux_inputs
    
    @staticmethod
    def init_from_config(config, samples: Union[List[List], List[Dict]] = None):
        """
        Initialize YourTTS model from config.
        This ensures we return a YourTTS instance, not a base VITS instance.
        """
        # Call parent's init_from_config which does all the setup
        model = Vits.init_from_config(config, samples)
        
        # Now we need to change the class of the returned object
        # This is a bit hacky but necessary since init_from_config is static
        model.__class__ = YourTTS
        
        return model
