import numpy as np
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


class VitsCustom(Vits):
    """
    Custom Vits model that correctly overrides both `format_batch` and `init_from_config`
    to ensure the model works with per-speaker d-vectors.
    """

    def format_batch(self, batch: dict) -> dict:
        """
        Correctly formats the batch by loading d-vectors using speaker_names.
        """
        # --- D-Vector Logic (The Fix) ---
        if (
            self.args.use_d_vector_file
            and self.speaker_manager
            and self.speaker_manager.embeddings
        ):
            d_vector_mapping = self.speaker_manager.embeddings
            # Use speaker_names, which matches the keys in your speakers.pth
            embedding_list = [
                d_vector_mapping[s]['embedding']
                for s in batch['speaker_names']
            ]

            d_vectors_as_tensors = [
                torch.from_numpy(e).float()
                if isinstance(e, np.ndarray)
                else torch.tensor(e, dtype=torch.float32)
                for e in embedding_list
            ]
            model_device = next(self.parameters()).device
            d_vectors = torch.stack(d_vectors_as_tensors).to(model_device)
            batch['d_vectors'] = d_vectors
        else:
            batch['d_vectors'] = None

        # --- Speaker ID Logic ---
        if self.speaker_manager and self.args.use_speaker_embedding:
            speaker_ids = [
                self.speaker_manager.name_to_id[s]
                for s in batch['speaker_names']
            ]
            batch['speaker_ids'] = torch.IntTensor(speaker_ids).to(
                next(self.parameters()).device
            )
        else:
            batch['speaker_ids'] = None

        # --- Language ID Logic ---
        if self.language_manager and self.args.use_language_embedding:
            language_ids = [
                self.language_manager.name_to_id[language]
                for language in batch['language_names']
            ]
            batch['language_ids'] = torch.LongTensor(language_ids).to(
                next(self.parameters()).device
            )
        else:
            batch['language_ids'] = None

        return batch

    @staticmethod
    def init_from_config(config: VitsConfig, samples: list = None):
        """
        (The Main Fix) This method overrides the original staticmethod to ensure
        it returns an instance of THIS class (VitsCustom) instead of the base Vits class.
        This version is also updated to correctly handle the config object.
        """
        ap = AudioProcessor.init_from_config(config)

        # Correctly re-assign config after tokenizer initialization, as in the base class
        tokenizer, config = TTSTokenizer.init_from_config(config)

        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path,
                config.model_args.speaker_encoder_config_path,
            )

        # Return an instance of `VitsCustom` using the potentially updated config
        return VitsCustom(
            config, ap, tokenizer, speaker_manager, language_manager
        )
