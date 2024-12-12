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
        paddings (list): List of paddings.
        dropout (float): The dropout rate.
    """

    def __init__(self, channels: list, kernel_sizes: list, paddings: list, dropout: float = 0.5):
        super(CNNClassifier, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=kernel_sizes[0], stride=1, padding=paddings[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_sizes[0], stride=1, padding=paddings[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=1, padding=paddings[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_sizes[1], stride=1, padding=paddings[1])
        
        # Global average pooling
        self.ada_avg = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(channels[1], channels[1] // 2)
        self.fc_relu1 = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(channels[1] // 2, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        
        # Average pooling layer
        x = self.ada_avg(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
