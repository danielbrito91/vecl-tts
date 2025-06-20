import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_melspectrogram_with_f0(wav_path: str, fmax_display: int = 2000):
    """
    Computes and plots a Mel spectrogram for a given audio file and overlays
    its fundamental frequency (F0) contour.

    This function is designed to make the F0 contour clearly visible, even for
    low pitch values, by using a Mel scale for the frequency axis.

    Args:
        wav_path (str): The file path to the input .wav audio file.
    """
    try:
        # 1. Load the audio file
        y, sr = librosa.load(wav_path, sr=None)

        # 2. Estimate the fundamental frequency (F0) using the pyin algorithm
        # fmin and fmax are set to common vocal ranges to improve accuracy
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )

        # Get the time points for the F0 plot
        times = librosa.times_like(f0, sr=sr)

        # 3. Compute the Mel spectrogram
        D = librosa.feature.melspectrogram(y=y, sr=sr)

        # Convert the power spectrogram to a decibel (dB) scale
        S_db = librosa.power_to_db(D, ref=np.max)

        # 4. Plot the spectrogram and the F0 contour
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the Mel spectrogram using a Mel-frequency scale on the y-axis
        # This scale enhances the visibility of lower frequencies
        img = librosa.display.specshow(
            S_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            fmax=fmax_display,  # Show frequencies up to 8kHz
            ax=ax,
        )

        # Add a color bar to show the intensity in dB
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity (dB)')

        # Overlay the F0 contour.
        # We only plot the points where the frame is considered "voiced"
        # The 'cyan' color and marker size are chosen for high visibility
        ax.scatter(
            times[voiced_flag],
            f0[voiced_flag],
            c='cyan',
            s=15,  # Size of the dots
            label='F0 Contour (voiced)',
            zorder=5,  # Plot on top of the spectrogram
        )

        # 5. Set plot titles and labels for clarity
        ax.set_title('Mel Spectrogram with F0 Pitch Contour', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f'An error occurred: {e}')
