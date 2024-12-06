"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import argparse
import numpy as np
import scipy.io.wavfile as wav

from scipy.signal import resample


def downsample_wav(input_wav: str, output_wav: str, target_rate: int) -> None:
    """
    Downsample a .wav file from 300kHz to the target rate.

    Args:
        input_wav (str): Path to the input .wav file.
        output_wav (str): Path to save the downsampled .wav file.
        target_rate (int): Target sample rate for the downsampled .wav file.
    """

    # Read the original .wav file
    sample_rate, data = wav.read(input_wav)

    # Check if the original sample rate is 300kHz
    if sample_rate != 300000:
        raise ValueError(f"Expected sample rate of 300kHz, but got {sample_rate}Hz.")

    # Calculate the number of samples for the new rate
    num_samples = int(len(data) * target_rate / sample_rate)

    # Resample the audio wav_data
    downsampled_data = resample(data, num_samples)
    downsampled_data = np.int16(data / np.max(np.abs(downsampled_data)) * 32767)

    # Write the downsampled wav_data to the new .wav file
    wav.write(output_wav, target_rate, downsampled_data.astype(np.int16))

    print(f"The downsampled file has been saved as {output_wav}")


def resample_in_batch(input_path: str, output_path: str, target_rate: int) -> None:
    """
    Downsample all .wav files in the input directory and save them in the output directory.

    Args:
        input_path (str): Path to the directory containing input .wav files.
        output_path (str): Path to save the downsampled .wav files.
        target_rate (int): Target sample rate for the downsampled .wav files.
    """

    # Get the list of .wav files in the input directory
    input_filelist = [x for x in os.listdir(input_path) if x.endswith('.wav')]

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Downsample each .wav file in the input directory
    for item in input_filelist:
        downsample_wav(os.path.join(input_path, item),
                       os.path.join(output_path, item),
                       target_rate)


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Downsample WAV files from 300kHz to 16kHz.")
    parser.add_argument('--input_path', type=str, default="./USV_Data",
                        required=True, help="Path to the directory containing the MAD-UV Dataset.")
    parser.add_argument('--output_path', type=str, default="./USV_Data_DS",
                        required=True, help="Path to save the downsampled MAD-UV Dataset.")
    parser.add_argument('--target_rate', type=int, default=16000,
                        help="Target sample rate for the downsampled .wav files.")
    args = parser.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Downsample all .wav files in the input directory
    for folder in ['train', 'valid', 'test']:
        resample_in_batch(os.path.join(args.input_path, folder),
                          os.path.join(args.output_path, folder),
                          args.target_rate)
