from pathlib import Path

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
