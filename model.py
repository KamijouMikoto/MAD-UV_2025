"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    A simple CNN-based classifier.

    Args:
        channels (list): List of numbers of channels.
        kernel_sizes (list): List of sizes of kernels.
        strides (list): List of strides.
        dropout (float): The dropout rate.
    """

    def __init__(self, channels: list, kernel_sizes: list, strides: list, dropout: float = 0.5):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, channels[0],
                               kernel_size=kernel_sizes[0], stride=strides[0], padding=(1, 1))
        self.conv2 = nn.Conv2d(channels[0], channels[1],
                               kernel_size=kernel_sizes[1], stride=strides[1], padding=(1, 1))
        self.conv3 = nn.Conv2d(channels[1], channels[2],
                               kernel_size=kernel_sizes[2], stride=strides[2], padding=(1, 1))
        
        # Global average pooling
        self.ada_avg = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(channels[2], channels[0])
        self.fc2 = nn.Linear(channels[0], 1)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.unsqueeze(1)

        # Convolutional layers
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        
        # Global average pooling
        x = self.ada_avg(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x
