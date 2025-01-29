"""
MADUV Challenge 2025 Baseline Code
Website: https://www.maduv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import scipy
import argparse
import numpy as np


def normalise(input_array: np.ndarray, norm_type: str = 'minmax') -> np.ndarray:
    """
    Normalise the input array using Min-Max Normalisation or Mean-std Standardisation.

    Args:
        input_array (np.ndarray): Input array to be normalised.
        norm_type (str): Normalisation method. Can be 'minmax', 'meanstd' or None.

    Returns:
        np.ndarray: Normalised array.
    """

    if norm_type == 'minmax':
        # Min-Max Normalisation
        min_val = np.min(input_array)
        max_val = np.max(input_array)
        # Avoid division by zero in case of constant values
        if max_val == min_val:
            print("Max and min value are same, skipping with all-zero array.")
            return np.zeros_like(input_array)
        else:
            # Apply min-max normalisation
            return (input_array - min_val) / (max_val - min_val)

    elif norm_type == 'meanstd':
        # Mean-std Standardisation
        mean_val = np.mean(input_array)
        std_val = np.std(input_array)
        # Avoid division by zero in case of constant values
        if std_val == 0:
            print("Standard deviation is zero, skipping with all-zero array.")
            return np.zeros_like(input_array)
        else:
            # Apply Mean-std Standardisation
            return (input_array - mean_val) / std_val
    else:
        return input_array


def average_spectrogram(freq_array: np.ndarray, t: np.ndarray, spec: np.ndarray, start_freq: int,
                        end_freq: int, bin_size: int, normalisation: str = 'minmax') -> np.ndarray:
    """
    Average the spectrogram over frequency bins.

    Args:
        freq_array (np.ndarray): The frequency array acquired from the scipy.signal.spectrogram().
        t (np.ndarray): The time array acquired from the scipy.signal.spectrogram().
        spec (np.ndarray): The spectrogram acquired from the scipy.signal.spectrogram().
        start_freq (int): The starting frequency.
        end_freq (int): The ending frequency.
        bin_size (int): The size of frequency bins.
        normalisation (str): Normalisation method. Can be 'minmax', 'meanstd' or None.

    Returns:
        np.ndarray: (Normalised) Averaged log-spectrogram.
    """

    # Define the frequency bins and construct the averaged spectrogram
    freq_bins = np.arange(start_freq, end_freq, bin_size)
    averaged_spec = np.zeros((len(freq_bins), len(t)))

    # Average the spectrogram over frequency bins
    for i, f in enumerate(freq_bins[:-1]):
        mask = (freq_array >= f) & (freq_array < f + bin_size)
        averaged_spec[i, :] = np.mean(spec[mask, :], axis=0)

    # Log-transform (and normalise) the averaged spectrogram
    averaged_spec = np.log10(1 + averaged_spec.T)
    averaged_spec = normalise(averaged_spec, normalisation)

    return averaged_spec


def extract_spectrogram(audio_file: str, save_file: str, n_fft: int = 300000, win_length: int = 300000,
                        hop_length: int = 150000, average: list = None, normalisation: str = 'minmax') -> None:
    """
    Extract a Spectrogram from an audio file and save it as a .npy file.

    Args:
        audio_file (str): Path to the input audio file.
        save_file (str): Path to save the extracted spectrogram.
        n_fft (int): Number of FFT components.
        win_length (int or None): Window size (in frame) for STFT.
        hop_length (int): Hop length (in frame) for STFT.
        average (list[int]): Parameters for averaging the spectrogram.
        normalisation (str): Normalisation method. Can be 'minmax', 'meanstd' or None.
    """

    sample_rate, audio_data = scipy.io.wavfile.read(audio_file)

    frequencies, times, spectrogram = scipy.signal.spectrogram(audio_data, fs=sample_rate, nperseg=win_length,
                                                               noverlap=hop_length, nfft=n_fft, mode='magnitude')

    # Spectrogram extraction and saving
    spec_full = average_spectrogram(frequencies, times, spectrogram,
                                    average[0], average[1], average[2], normalisation)
    np.save(save_file, spec_full)
    print(f"Saved full spectrogram to {save_file}, shape: {spec_full.shape}")
    

def process_audio_files(input_folder: str, output_folder: str, feature_set: str) -> None:
    """
    Process audio files in the input folder and save the extracted features in the output folder.

    Args:
        input_folder (str): Path to the directory containing input audio files.
        output_folder (str): Path to the directory saving the extracted features.
        feature_set (str): The selected feature set. Can be 'full', 'ultra' or 'audi'.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each audio file in the input directory
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.npy")

            if feature_set == 'full':
                extract_spectrogram(audio_path, output_path, average=[0, 150000, 300])
            elif feature_set == 'ultra':
                extract_spectrogram(audio_path, output_path, average=[20000, 150000, 260])
            else:
                extract_spectrogram(audio_path, output_path, average=[0, 20000, 40])


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Extract traditional feature sets.")
    parser.add_argument('--input_path', type=str, default="./USV_Data",
                        required=True, help="Path to the directory containing input wav files.")
    parser.add_argument('--output_path', type=str, default="./USV_Feature_full",
                        required=True, help="Path to save the extracted feature files.")
    parser.add_argument('--feature_set', type=str, default="full",
                        required=True, help="Selected feature set, including full, ultra and audi.")
    args = parser.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Process all audio files in the input directory
    for folder in ['train', 'valid', 'test']:
        if args.feature_set not in ['full', 'ultra', 'audi']:
            print("feature_set can only be 'full', 'ultra' or 'audi'.")
            break
        else:
            process_audio_files(os.path.join(args.input_path, folder),
                                os.path.join(args.output_path, folder),
                                args.feature_set)
