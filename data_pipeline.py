"""
MADUV Challenge 2025 Baseline Code
Website: https://www.maduv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import torch
import torch.nn
import numpy as np

from torch.utils.data import Dataset


# Custom Dataset for loading feature files
class AudioFeatureDataset(Dataset):
    """
    A custom Dataset for loading feature files.

    Args:
        data_folder (str): Path to the directory containing feature files.
        length (int): The length of feature vectors.
        is_test (bool): Whether the dataset is for testing.
    """

    def __init__(self, data_folder: str, length: int, is_test: bool = False):
        self.data_folder = data_folder
        self.file_list = [x for x in os.listdir(data_folder) if x.endswith(".npy")]
        self.length = length
        self.is_test = is_test

        # Sort the file list for consistency, especially for testing
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the feature file path
        feature_path = os.path.join(self.data_folder, self.file_list[idx])

        # Load the feature vectors
        features = np.load(feature_path)

        # Padding with 0
        if features.shape[0] != self.length:
            padding = self.length - features.shape[0]
            features = np.pad(features, pad_width=((0, padding), (0, 0)), mode='constant')
        features = torch.tensor(features, dtype=torch.float32)
        
        # Get sample name
        sample_name = self.file_list[idx]

        # No need to prepare the label for test set
        if self.is_test:
            return features
        
        # Extract the label from the file name (assuming file names are like 'classX_filename.npy')
        label = int(self.file_list[idx].split('-')[-1][0])
        label = torch.tensor(label, dtype=torch.long)

        return features, label, sample_name.split('-')[0]
