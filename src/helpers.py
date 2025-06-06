"""
Helper functions for the project.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from typing import Literal
from src.dataset.dataset import TestDataset, TrainDataset
import tqdm
import numpy as np



def save_submission(surveys: list[int],
                    top_k_indices: list[list[int]],
                    filename: str
                    ) -> None:
    """
    Save the submission file.
    """
    data_concatenated = [' '.join(map(str, row)) for row in top_k_indices]
    pd.DataFrame(
        {
            'surveyId': surveys,
            'predictions': data_concatenated,
        }
    ).to_csv(filename, index = False)

def select_device() -> torch.device:
    """
    Select the device to use for training.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_loop(model: torch.nn.Module,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               device: torch.device = torch.device('cpu'),
               num_epochs: int = 25,
               *scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
               positive_weight_factor: float = 1.0
               ):
    print(f"Training for {num_epochs} epochs started.")

    for epoch in range(num_epochs):
        # set the model to training mode
        model.train()
        for batch_idx, (data, targets, _) in enumerate(train_loader):

            # move the data to the device
            data = data.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(data)

            # compute the loss
            pos_weight = targets*positive_weight_factor  # All positive weights are equal to 10
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if batch_idx % 348 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        if scheduler:
            scheduler.step()
            print("Scheduler:",scheduler.state_dict())





def test_loop(model: torch.nn.Module,
              test_loader: DataLoader,
              device: torch.device = torch.device('cpu'),
              ) -> tuple[list, np.ndarray]:
    """
    Test loop for the model.
    """
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        surveys = []
        top_k_indices = None
        # tqdm for progress bar
        for data, survey_id in tqdm.tqdm(test_loader, total=len(test_loader)):

            # Move data to device
            data = data.to(device)

            # Forward pass
            outputs = model(data)

            # Apply sigmoid to outputs to get probabilities
            predictions = torch.sigmoid(outputs).cpu().numpy()

            # Select top-25 values as predictions
            top_25 = np.argsort(-predictions, axis=1)[:, :25]
            if top_k_indices is None:
                # Initialize top_k_indices with top_25
                top_k_indices = top_25
            else:
                # Concatenate top_k_indices and top_25
                top_k_indices = np.concatenate((top_k_indices, top_25), axis=0)

            surveys.extend(survey_id.cpu().numpy())
    return surveys, top_k_indices
