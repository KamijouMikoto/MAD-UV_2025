"""
MAD-UV Challenge 2025 Baseline Code
Website: https://www.mad-uv.org/
Author: Zijiang YANG (The University of Tokyo), et al.
Date: November 2024
"""

import os
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import CNNClassifier
from hyperparameter import hparam
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from data_pipeline import AudioFeatureDataset


def train(net: CNNClassifier, data_loader: DataLoader) -> float:
    """
    Training process of the model.

    Args:
        net (CNNClassifier): The model to be trained.
        data_loader (DataLoader): DataLoader for the training set.

    Returns:
        float: The average loss of one training epoch.
    """

    net.train()
    total_loss = 0.

    for i, (data, y_true, _) in enumerate(data_loader):
        data = data.to(device)
        y_true = y_true.to(torch.float32).to(device)

        optimiser.zero_grad()
        output = net(data)
        loss = criterion(output, y_true.unsqueeze(-1))
        loss.backward()
        optimiser.step()

        print(f"Epoch {epoch+1:04} - Batch {i+1}/{len(data_loader)}\tLoss: {loss.item():.3f}")

        total_loss += loss.item()

    epoch_loss = total_loss / len(data_loader)
    return epoch_loss


def validate(net: CNNClassifier, data_loader: DataLoader) -> tuple:
    """
    Validation process of the model.

    Args:
        net (CNNClassifier): The model to be validated.
        data_loader (DataLoader): DataLoader for the validation set.

    Returns:
        tuple: A tuple containing the validation loss, the predicted probabilities, the true labels, and the filenames.
    """

    net.eval()
    total_loss = 0.
    all_probabilities = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for i, (data, y_true, y_name) in enumerate(data_loader):
            data = data.to(device)
            y_true = y_true.to(torch.float32).to(device)

            output = net(data)
            loss = criterion(output, y_true.unsqueeze(-1))

            print(f"vEpoch {epoch+1:04} - Batch {i+1}/{len(data_loader)}\tLoss: {loss.item():.3f}")

            total_loss += loss.item()

            y_prob = torch.sigmoid(output)

            all_probabilities.append(y_prob.squeeze())
            all_labels.append(y_true)
            all_filenames = all_filenames + list(y_name)

    epoch_loss = total_loss / len(data_loader)

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return epoch_loss, all_probabilities, all_labels, all_filenames


if __name__ == '__main__':
    # Set the seeds
    seed = hparam['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set the device
    device = torch.device(hparam['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the model
    model = CNNClassifier(channels=hparam['model']['channels'],
                          kernel_sizes=hparam['model']['kernel_sizes'],
                          paddings=hparam['model']['paddings'],
                          dropout=hparam['model']['dropout']).to(device)

    # Set the loss function and optimiser
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(),
                           lr=hparam['training']['learning_rate'],
                           weight_decay=hparam['training']['weight_decay'])

    # Set the training hyperparameters
    batch_size = hparam['training']['batch_size']
    training_epoch = hparam['training']['num_epochs']
    validation_epoch = hparam['training']['valid_after_epoch']

    # Set the tensorboard writer and log file
    writer = SummaryWriter(hparam['path']['tensorboard_path'])
    log_path = hparam['path']['log_path']

    # Create datasets and dataloaders for train and validation sets
    train_dataset = AudioFeatureDataset(data_folder=hparam['path']['train_path'],
                                        length=hparam['model']['feature_length'])
    valid_dataset = AudioFeatureDataset(data_folder=hparam['path']['valid_path'],
                                        length=hparam['model']['feature_length'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Training phase
    best_seg_uar = 0
    for epoch in range(training_epoch):
        # Get the training loss
        train_loss = train(net=model, data_loader=train_loader)

        # Print loss and write the log
        print(f"-->\tEpoch {epoch+1:04}:\tTraining Loss: {train_loss:.3f}")
        writer.add_scalar('loss/training', train_loss, epoch+1)
        with open(log_path, 'a') as lf:
            lf.write(f"-->\tEpoch {epoch+1:04}:\tTraining Loss: {train_loss:.3f}\n")

        # Validation phase
        if (epoch + 1) % validation_epoch == 0:
            # Get the validation loss, predicted probabilities, true labels, and filenames
            valid_loss, probabilities, labels, filenames = validate(net=model, data_loader=valid_loader)

            # Calculate the segment-level UAR by grid search the threshold
            max_uar = 0
            best_threshold = 0
            threshold = 0.1
            while threshold <= 0.9:
                predictions = (probabilities >= threshold).float()
                uar = recall_score(labels.detach().cpu(), predictions.detach().cpu(), average='macro')
                if uar >= max_uar:
                    max_uar = uar
                    best_threshold = threshold
                threshold += 0.05

            # Print loss and segment-level metric, and write the log
            print(f"-->\tEpoch {epoch+1:04}:\tValidation Loss: {valid_loss:.3f}")
            print(f"-->\tEpoch {epoch+1:04}:\tSegment UAR: {max_uar:.3f} at Threshold {best_threshold:.3f}")
            writer.add_scalar('loss/validation', valid_loss, epoch+1)
            writer.add_scalar('seg_metric/uar', max_uar, epoch+1)
            with open(log_path, 'a') as lf:
                lf.write(f"-->\tEpoch {epoch+1:04}:\tValidation Loss: {valid_loss:.3f}\n")
                lf.write(f"-->\tEpoch {epoch+1:04}:\tSegment UAR: {max_uar:.3f} "
                         f"at Threshold {best_threshold:.3f}\n")

            # Save the model if the segment-level UAR is improved
            if max_uar > best_seg_uar:
                best_seg_uar = max_uar
                model_path = os.path.join(hparam['path']['model_path'], f"best_segment_{best_threshold:.3f}.pth")
                
                if not os.path.exists(hparam['path']['model_path']):
                    os.makedirs(hparam['path']['model_path'])
                
                saved_model_list = glob.glob(os.path.join(hparam['path']['model_path'], "best_segment_*.pth"))
                for saved_model in saved_model_list:
                    os.remove(saved_model)
                    
                torch.save(model.state_dict(), model_path)
                print(f"Model saved at {model_path}")
                with open(log_path, 'a') as lf:
                    lf.write(f"Model saved at {model_path}\n")

            # Calculate the subject-level UAR with majority vote
            predictions = (probabilities >= best_threshold).float()

            grouped_predictions = defaultdict(list)
            grouped_labels = defaultdict(list)

            for pred, label, filename in zip(predictions, labels, filenames):
                grouped_predictions[filename].append(pred)
                grouped_labels[filename].append(label)

            final_predictions = []
            final_labels = []
            for filename in grouped_predictions:
                # Majority vote for predictions and labels
                majority_pred = max(set(grouped_predictions[filename]), key=grouped_predictions[filename].count)
                majority_label = max(set(grouped_labels[filename]), key=grouped_labels[filename].count)

                final_predictions.append(majority_pred.detach().cpu())
                final_labels.append(majority_label.detach().cpu())

            uar = recall_score(final_labels, final_predictions, average='macro')

            # Print the subject-level metric and write the log
            print(f"-->\tEpoch {epoch+1:04}:\tSubject UAR: {uar:.3f} at Threshold {best_threshold:.3f}")
            writer.add_scalar('sam_metric/uar', uar, epoch+1)
            with open(log_path, 'a') as lf:
                lf.write(f"-->\tEpoch {epoch+1:04}:\tSubject UAR: {uar:.3f} at Threshold {best_threshold:.3f}\n")
