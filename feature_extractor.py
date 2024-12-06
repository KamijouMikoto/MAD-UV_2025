"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import scipy
import torch
import argparse
import audiofile
import opensmile
import torchaudio
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


def extract_spec_ds(audio_file: str, save_file: str, n_fft: int = 1534,
                    win_length: int = 1534, hop_length: int = 320, normalisation: str = 'minmax') -> None:
    """
    Extract a Spectrogram from an audio file and save it as a .npy file.

    Args:
        audio_file (str): Path to the input audio file.
        save_file (str): Path to save the extracted spectrogram.
        n_fft (int): Number of FFT components.
        win_length (int or None): Window size (in frame) for STFT.
        hop_length (int): Hop length (in frame) for STFT.
        normalisation (str): Normalisation method. Can be 'minmax', 'meanstd' or None.
    """

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Check if the sample rate is 16kHz
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz sample rate, but got {sample_rate} Hz.")

    # Move waveform to the GPU if available
    waveform = waveform.to(device)

    # Create a Spectrogram transformation
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0  # Power spectrogram
    ).to(device)

    # Apply the transformation
    spec = spectrogram_transform(waveform)

    # Move back to CPU for saving
    spec = spec.squeeze(0).T.cpu().numpy()
    spec = np.log10(1 + spec)

    normalised_spec = normalise(spec, normalisation)

    # Save the spectrogram as a .npy file
    np.save(save_file, normalised_spec)
    print(f"Saved spectrogram to {save_file}, shape: {normalised_spec.shape}")


def extract_spec_a(audio_file: str, save_file: str, n_fft: int = 300000, win_length: int = 300000,
                   hop_length: int = 150000, bin_size: int = 500, normalisation: str = None):
    """
    Extract a Spectrogram from an audio file and save it as a .npy file.

    Args:
        audio_file (str): Path to the input audio file.
        save_file (str): Path to save the extracted spectrogram.
        n_fft (int): Number of FFT components.
        win_length (int or None): Window size (in frame) for STFT.
        hop_length (int): Hop length (in frame) for STFT.
        bin_size (int): Size of frequency bins for averaging.
        normalisation (str): Normalisation method. Can be 'minmax', 'meanstd' or None.
    """

    sample_rate, audio_data = scipy.io.wavfile.read(audio_file)

    frequencies, times, spectrogram = scipy.signal.spectrogram(audio_data, fs=sample_rate, nperseg=win_length,
                                                               noverlap=hop_length, nfft=n_fft, mode='magnitude')

    freq_bins = np.arange(0, frequencies[-1], bin_size)
    averaged_spectrogram = np.zeros((len(freq_bins), len(times)))

    for i, f in enumerate(freq_bins[:-1]):
        mask = (frequencies >= f) & (frequencies < f + bin_size)
        averaged_spectrogram[i, :] = np.mean(spectrogram[mask, :], axis=0)

    spec = np.log10(1 + averaged_spectrogram.T)

    normalised_spec = normalise(spec, normalisation)

    np.save(save_file, normalised_spec)
    print(f"Saved spectrogram to {save_file}, shape: {normalised_spec.shape}")


def extract_opensmile(audio_file: str, save_file: str, smile: opensmile.core.smile.Smile,
                      window_length: float = 0.2, hop_length: float = 0.1, normalisation: str = 'minmax') -> None:
    """
    Extract OpenSMILE features from an audio file and save it as a .npy file.

    Args:
        audio_file (str): Path to the input audio file.
        save_file (str): Path to save the extracted OpenSMILE features.
        smile (opensmile.core.smile.Smile): OpenSMILE object.
        window_length (float): Window size (in second) for sliding window.
        hop_length (float): Hop length (in second) for sliding window.
        normalisation (str): Normalisation method. Can be 'minmax', 'meanstd' or None.
    """

    # Load the audio file
    signal, sr = audiofile.read(audio_file, always_2d=False)

    # Extract OpenSMILE features using sliding window
    features = []
    for start in range(0, len(signal), int(hop_length*sr)):
        end = start + int(window_length * sr)
        if end > len(signal):
            break
        else:
            window = signal[start:end]
        
        feature = smile.process_signal(window, sr).to_numpy()
        features.append(feature)
    features = np.stack(features).squeeze()

    if normalisation == 'minmax':
        # Min-Max Normalisation along the feature axis
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero in case of constant values
        range_vals[range_vals == 0] = 1
        # Apply normalisation
        normalised_features = (features - min_vals) / range_vals

    elif normalisation == 'meanstd':
        # Mean-std Standardisation along the feature axis
        mean_vals = np.mean(features, axis=0)
        std_vals = np.std(features, axis=0)
        # Avoid division by zero in case of constant values
        std_vals[std_vals == 0] = 1
        # Apply normalisation
        normalised_features = (features - mean_vals) / std_vals

    else:
        normalised_features = features

    # Save the OpenSMILE feature as a .npy file
    np.save(save_file, normalised_features)
    print(f"Saved OpenSMILE to {save_file}, shape: {normalised_features.shape}")
    

def process_audio_files(input_folder: str, output_folder: str, feature_set: str) -> None:
    """
    Process audio files in the input folder and save the extracted features in the output folder.

    Args:
        input_folder (str): Path to the directory containing input audio files.
        output_folder (str): Path to the directory saving the extracted features.
        feature_set (str): The selected feature set. Can be 'spec_ds', 'spec_a' or 'egemaps'.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each audio file in the input directory
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.npy")

            if feature_set == 'spec_ds':
                extract_spec_ds(audio_path, output_path)
            elif feature_set == 'spec_a':
                extract_spec_a(audio_path, output_path)
            else:
                # Create an OpenSMILE object
                smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                        feature_level=opensmile.FeatureLevel.Functionals)
                extract_opensmile(audio_path, output_path, smile)


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Extract traditional feature sets.")
    parser.add_argument('--input_path', type=str, default="./USV_Data_DS",
                        required=True, help="Path to the directory containing input wav files.")
    parser.add_argument('--output_path', type=str, default="./USV_Feature_spec",
                        required=True, help="Path to save the extracted feature files.")
    parser.add_argument('--feature_set', type=str, default="egemaps",
                        required=True, help="Selected feature set, including spec_ds, spec_a or egemaps.")
    args = parser.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # Set the device for spectrogram extraction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process all audio files in the input directory
    for folder in ['train', 'valid', 'test']:
        if args.feature_set not in ['egemaps', 'spec_ds', 'spec_a']:
            print("feature_set can only be 'egemaps', 'spec_ds' or 'spec_a'.")
            break
        else:
            process_audio_files(os.path.join(args.input_path, folder),
                                os.path.join(args.output_path, folder),
                                args.feature_set)
