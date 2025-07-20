import os
import shutil
import subprocess
import tempfile
from glob import glob

import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm import tqdm


class AudioPreprocessor:
    """Audio preprocessing pipeline using VAD and ffmpeg-normalize."""

    def __init__(self, target_lufs: float, sampling_rate: int):
        self.vad_model = load_silero_vad()
        self.target_lufs = target_lufs
        self.sampling_rate = sampling_rate

        if not self._check_ffmpeg_normalize():
            raise RuntimeError(
                'ffmpeg-normalize not found. Please install it.'
            )

    @staticmethod
    def _check_ffmpeg_normalize():
        """Check if ffmpeg-normalize is available."""
        try:
            subprocess.run(
                ['ffmpeg-normalize', '--help'], capture_output=True, check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def remove_silence(self, wav: torch.Tensor) -> torch.Tensor:
        """Remove silence from beginning and end of audio using VAD."""
        if wav.numel() == 0:
            return wav

        speech_timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=self.sampling_rate,
            return_seconds=False,
        )
        if not speech_timestamps:
            return torch.tensor([], dtype=wav.dtype, device=wav.device)

        start_sample = speech_timestamps[0]['start']
        end_sample = speech_timestamps[-1]['end']
        return wav[start_sample:end_sample]

    def normalize_with_ffmpeg(self, input_file: str, output_file: str):
        """Normalize audio using ffmpeg-normalize."""
        cmd = [
            'ffmpeg-normalize',
            input_file,
            '-o',
            output_file,
            '-f',  # Force overwrite
            '-nt',
            'rms',
            '-t',
            str(self.target_lufs),
            '--keep-loudness-range-target',
            '--print-stats',
        ]
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f'ffmpeg-normalize failed: {e.stderr}'

    def process_single_file(
        self, audio_file: str, output_file: str, temp_dir: str = None
    ):
        """Process a single audio file with VAD + ffmpeg normalization."""
        cleanup_temp = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            cleanup_temp = True

        try:
            wav = read_audio(audio_file, sampling_rate=self.sampling_rate)
            wav_processed = self.remove_silence(wav)

            if wav_processed.numel() == 0:
                return False, 'Empty audio after VAD'

            temp_filename = os.path.join(
                temp_dir, f'temp_{os.path.basename(audio_file)}'
            )
            if wav_processed.ndim == 1:
                wav_processed = wav_processed.unsqueeze(0)
            torchaudio.save(temp_filename, wav_processed, self.sampling_rate)

            success, message = self.normalize_with_ffmpeg(
                temp_filename, output_file
            )
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return success, message
        except Exception as e:
            return False, f'Error processing {audio_file}: {str(e)}'
        finally:
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        file_pattern: str = '*.wav',
    ):
        """Process all audio files in a folder."""
        os.makedirs(output_folder, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix='audio_preprocessing_')
        try:
            audio_files = glob(os.path.join(input_folder, file_pattern))
            if not audio_files:
                print(f"No files matching '{file_pattern}' in {input_folder}")
                return []

            processed_files, failed_files = [], []
            for audio_file in tqdm(audio_files, desc='Processing audio files'):
                output_filename = os.path.join(
                    output_folder, os.path.basename(audio_file)
                )
                success, message = self.process_single_file(
                    audio_file, output_filename, temp_dir
                )
                if success:
                    processed_files.append(output_filename)
                else:
                    failed_files.append((audio_file, message))

            print(
                f'\nSuccessfully processed: {len(processed_files)}, Failed: {len(failed_files)}'
            )
            for file, error in failed_files:
                print(f'  - {os.path.basename(file)}: {error}')
            return processed_files
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
