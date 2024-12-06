"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import re
import glob
import torch

from torch.utils.data import DataLoader
from hyperparameter import hparam, feature
from data_pipeline import AudioFeatureDataset
from model import CNNClassifier


def test(net: CNNClassifier, data_loader: DataLoader) -> torch.Tensor:
    """
    Generate predicted probabilities of the test set.

    Args:
        net (CNNClassifier): The model to be tested.
        data_loader (DataLoader): DataLoader for the test set.

    Returns:
        tuple: A tuple containing the predicted probabilities and filenames.
    """
    net.eval()
    all_probabilities = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output = net(data)

            y_prob = torch.sigmoid(output)
            all_probabilities.append(y_prob.squeeze())

    all_probabilities = torch.cat(all_probabilities, dim=0)

    return all_probabilities


if __name__ == '__main__':
    # Set the device
    device = torch.device(hparam['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader for test set
    test_dataset = AudioFeatureDataset(data_folder=hparam['path']['test_path'], 
                                       length=hparam['model']['feature_length'], 
                                       is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=hparam['training']['batch_size'], shuffle=False)

    # Load the model
    model = CNNClassifier(channels=hparam['model']['channels'],
                          kernel_sizes=hparam['model']['kernel_sizes'],
                          strides=hparam['model']['strides'],
                          dropout=hparam['model']['dropout']).to(device)
    segment_model = glob.glob(os.path.join(hparam['path']['model_path'], "best_segment_*.pth"))
    model.load_state_dict(torch.load(segment_model[0]))
    model = model.to(device)

    # Test phase
    probabilities = test(net=model, data_loader=test_loader)
    
    threshold = float(re.search("best_segment_(.*).pth", segment_model[0]).group(1))

    predictions = (probabilities >= threshold).float().detach().cpu().numpy().astype(int)
    
    results = ",".join(map(str, predictions))
    with open(f"./prediction_{feature}_segment.txt", 'w') as pf:
        pf.write(results)
    
    print(f"The segment-level prediction of {feature} has been saved to ./prediction_{feature}_segment.txt")
