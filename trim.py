"""
MADUV Challenge 2025 Baseline Code
Website: https://www.maduv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import argparse

from pydub import AudioSegment


def trim_audio_with_overlap(input_file: str, output_folder: str,
                            chunk_length: int = 30 * 1000, overlap_length: int = 15 * 1000):
    """
    Trim the .wav file according to the chunk length and overlap length.

    Args:
        input_file (str): Path to the input .wav file.
        output_folder (str): Path to the directory containing the trimmed .wav file.
        chunk_length (int): The length of chunks in millisecond.
        overlap_length (int): The length of overlap in millisecond.
    """

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize start position
    start = 0
    end = chunk_length
    count = 0

    # Loop to create audio chunks
    while end < len(audio):
        # Extract the chunk of audio
        chunk = audio[start:end]

        # Define the output file path
        id_type = input_file.split('/')[-1].split('-')[0]
        label = input_file.split('/')[-1][-5]
        chunk_filename = os.path.join(output_folder, f"{id_type}-chunk_{count}-{label}.wav")

        # Export the chunk
        chunk.export(chunk_filename, format="wav")
        print(f"Saved {chunk_filename}")

        # Move the window by the overlap amount
        start = start + (chunk_length - overlap_length)
        end = start + chunk_length

        # Increment chunk counter
        count += 1


def trim_in_batch(input_folder: str, output_folder: str, chunk_length: int, overlap_length: int) -> None:
    """
    Trim all .wav files in the input directory and save them in the output directory.

    Args:
        input_folder (str): Path to the directory containing input .wav files.
        output_folder (str): Path to save the trimmed .wav files.
        chunk_length (int): The length of the trimmed .wav files.
        overlap_length (int): The length of the overlap when trimming.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each audio file in the input directory
    input_filelist = [x for x in os.listdir(input_folder) if x.endswith('.wav')]
    for item in input_filelist:
        item = os.path.join(input_folder, item)
        trim_audio_with_overlap(item, output_folder, chunk_length, overlap_length)


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Trim audio sample.")
    parser.add_argument('--input_path', type=str, default="./USV_Data",
                        required=True, help="Path to the directory containing the MAD-UV Dataset.")
    parser.add_argument('--output_path', type=str, default="./USV_Data_DS",
                        required=True, help="Path to save the trimmed MAD-UV Dataset.")
    parser.add_argument('--chunk', type=int, default=30000,
                        help="The chunk length of trimmed audio in millisecond.")
    parser.add_argument('--overlap', type=int, default=15000,
                        help="The overlap length of trimmed audio in millisecond.")
    args = parser.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Trim all .wav files in the input directory
    for folder in ['train', 'valid']:
        trim_in_batch(os.path.join(args.input_path, folder),
                      os.path.join(args.output_path, folder),
                      args.chunk, args.overlap)
