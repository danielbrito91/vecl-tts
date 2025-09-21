import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_melspectrogram_with_f0(
    wav_path: str, fmax_display: int = 2000, output_path: str = None
):
    """
    Computes and plots a Mel spectrogram for a given audio file and overlays
    its fundamental frequency (F0) contour.

    This function is designed to make the F0 contour clearly visible, even for
    low pitch values, by using a Mel scale for the frequency axis.

    Args:
        wav_path (str): The file path to the input .wav audio file.
        fmax_display (int): Maximum frequency to display in Hz. Defaults to 2000.
        output_path (str, optional): Path to save the plot. If None, the plot will be displayed instead.
    """
    try:
        # Validate input parameters
        if not isinstance(wav_path, str):
            raise ValueError(
                f'wav_path must be a string, got {type(wav_path)}'
            )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f'Audio file not found: {wav_path}')

        if not isinstance(fmax_display, (int, float)) or fmax_display <= 0:
            raise ValueError(
                f'fmax_display must be a positive number, got {fmax_display}'
            )

        # 1. Load the audio file
        print(f'Loading audio file: {wav_path}')
        y, sr = librosa.load(wav_path, sr=None)

        # Validate audio data
        if y is None or len(y) == 0:
            raise ValueError(
                'Failed to load audio data or audio file is empty'
            )

        if sr is None or sr <= 0:
            raise ValueError(f'Invalid sample rate: {sr}')

        print(f'Audio loaded successfully: {len(y)} samples at {sr} Hz')

        # 2. Estimate the fundamental frequency (F0) using the pyin algorithm
        # fmin and fmax are set to common vocal ranges to improve accuracy
        print('Computing F0 contour...')
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )

        # Validate F0 output
        if f0 is None:
            raise ValueError('F0 estimation failed')

        print(
            f'F0 computed: {len(f0)} frames, {np.sum(voiced_flag)} voiced frames'
        )

        # Get the time points for the F0 plot
        times = librosa.times_like(f0, sr=sr)

        if times is None or len(times) != len(f0):
            raise ValueError('Failed to generate time array for F0')

        # 3. Compute the Mel spectrogram
        print('Computing Mel spectrogram...')
        D = librosa.feature.melspectrogram(y=y, sr=sr)

        if D is None or D.size == 0:
            raise ValueError('Failed to compute Mel spectrogram')

        # Convert the power spectrogram to a decibel (dB) scale
        S_db = librosa.power_to_db(D, ref=np.max)

        if S_db is None or S_db.size == 0:
            raise ValueError('Failed to convert spectrogram to dB scale')

        print(f'Spectrogram computed: shape {S_db.shape}')

        # 4. Plot the spectrogram and the F0 contour
        print('Creating plot...')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the Mel spectrogram using a Mel-frequency scale on the y-axis
        # This scale enhances the visibility of lower frequencies
        img = librosa.display.specshow(
            S_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            fmax=fmax_display,
            ax=ax,
        )

        # Add a color bar to show the intensity in dB
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity (dB)')

        # Overlay the F0 contour.
        # We only plot the points where the frame is considered "voiced"
        # The 'cyan' color and line width are chosen for high visibility
        voiced_times = times[voiced_flag]
        voiced_f0 = f0[voiced_flag]

        if len(voiced_times) > 0 and len(voiced_f0) > 0:
            ax.plot(
                voiced_times,
                voiced_f0,
                color='cyan',
                linewidth=2,  # Line thickness
                label='F0 Contour (voiced)',
                zorder=5,  # Plot on top of the spectrogram
            )

        # 5. Set plot titles and labels for clarity
        ax.set_title('Mel Spectrogram with F0 Pitch Contour', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.legend()
        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f'Plot saved to: {output_path}')
        else:
            plt.show()

        return fig

    except Exception as e:
        print(f'An error occurred: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback

        traceback.print_exc()
        return None
