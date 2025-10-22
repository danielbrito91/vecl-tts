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
    """Audio preprocessing pipeline for TTS using VAD and ffmpeg-normalize"""

    def __init__(self, target_lufs=-27.0, sampling_rate=16000):
        self.vad_model = load_silero_vad()
        self.target_lufs = target_lufs
        self.sampling_rate = sampling_rate

        if not self._check_ffmpeg_normalize():
            raise RuntimeError(
                'ffmpeg-normalize not found. Please install it.'
            )

    @staticmethod
    def _check_ffmpeg_normalize():
        """Check if ffmpeg-normalize is available in the system"""
        try:
            subprocess.run(
                ['ffmpeg-normalize', '--help'], capture_output=True, check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def remove_silence(self, wav):
        """Remove silence from beginning and end of audio using VAD"""
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

        # Keep audio from the start of the first speech segment to the end of the last one
        start_sample = speech_timestamps[0]['start']
        end_sample = speech_timestamps[-1]['end']
        return wav[start_sample:end_sample]

    def normalize_with_ffmpeg(self, input_file, output_file):
        """Normalize audio using ffmpeg-normalize to match VECL-TTS preprocessing"""
        cmd = [
            'ffmpeg-normalize',
            input_file,
            '-o',
            output_file,
            '-f',  # Force overwrite
            '-nt',
            'rms',  # Use RMS normalization type (matching VECL-TTS)
            '-t',
            str(self.target_lufs),  # Target loudness
            '--keep-loudness-range-target',  # Preserve dynamic range
            '--print-stats',  # Print normalization stats
        ]

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f'ffmpeg-normalize failed: {e.stderr}'
            return False, error_msg

    def process_single_file(self, audio_file, output_file, temp_dir=None):
        """Process a single audio file with VAD + ffmpeg normalization"""

        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            cleanup_temp = True
        else:
            cleanup_temp = False

        try:
            # Step 1: Load audio and apply VAD
            print(f'Applying VAD to {os.path.basename(audio_file)}...')
            wav = read_audio(audio_file, sampling_rate=self.sampling_rate)
            wav_processed = self.remove_silence(wav)

            if wav_processed.numel() == 0:
                print(
                    f'Warning: Audio {audio_file} is empty after VAD. Skipping.'
                )
                return False, 'Empty audio after VAD'

            # Step 2: Save to temporary file
            temp_filename = os.path.join(
                temp_dir, f'temp_{os.path.basename(audio_file)}'
            )

            # Ensure wav_processed has correct dimensions for torchaudio.save
            if wav_processed.ndim == 1:
                wav_processed = wav_processed.unsqueeze(0)

            torchaudio.save(temp_filename, wav_processed, self.sampling_rate)

            # Step 3: Apply ffmpeg normalization
            print(f'Normalizing {os.path.basename(audio_file)}...')
            success, message = self.normalize_with_ffmpeg(
                temp_filename, output_file
            )

            # Step 4: Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            return success, message

        except Exception as e:
            return False, f'Error processing {audio_file}: {str(e)}'

        finally:
            # Clean up temporary directory if we created it
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def process_folder(
        self, input_folder, output_folder, file_pattern='*.wav'
    ):
        """Process all audio files in a folder"""

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix='audio_preprocessing_')

        try:
            # Find all audio files
            audio_files = glob(os.path.join(input_folder, file_pattern))

            if not audio_files:
                print(
                    f"No audio files matching '{file_pattern}' found in {input_folder}"
                )
                return []

            print(f'Found {len(audio_files)} audio files to process.')

            processed_files = []
            failed_files = []

            for audio_file in tqdm(audio_files, desc='Processing audio files'):
                output_filename = os.path.join(
                    output_folder, os.path.basename(audio_file)
                )

                success, message = self.process_single_file(
                    audio_file, output_filename, temp_dir
                )

                if success:
                    processed_files.append(output_filename)
                    print(
                        f'✓ Successfully processed: {os.path.basename(audio_file)}'
                    )
                else:
                    failed_files.append((audio_file, message))
                    print(
                        f'✗ Failed to process: {os.path.basename(audio_file)} - {message}'
                    )

            # Print summary
            print('\n=== Processing Summary ===')
            print(f'Total files: {len(audio_files)}')
            print(f'Successfully processed: {len(processed_files)}')
            print(f'Failed: {len(failed_files)}')

            if failed_files:
                print('\nFailed files:')
                for failed_file, error in failed_files:
                    print(f'  - {os.path.basename(failed_file)}: {error}')

            return processed_files

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    """Main function to process audio files"""
    # Configuration
    input_folder = 'data/raw02/audio/'
    output_folder = 'data/processed02/audio/'
    target_lufs = -27.0  # Same as VECL-TTS

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        print('Please create the folder and add your .wav files.')
        return

    # Initialize preprocessor
    try:
        preprocessor = AudioPreprocessor(target_lufs=target_lufs)
    except RuntimeError as e:
        print(f'Error: {e}')
        return

    # Process all files
    processed_files = preprocessor.process_folder(input_folder, output_folder)

    if processed_files:
        print(
            f'\nProcessing complete! Processed files saved to: {output_folder}'
        )
    else:
        print('\nNo files were successfully processed.')


if __name__ == '__main__':
    main()
