"""
Module for creating data loaders.
"""

from typing import Literal
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from .dataset import TrainDataset, TestDataset
from .helpers import construct_patch_path, quantile_normalize

def create_data_loader(dataset_path: str,
                       split: Literal["train", "test"] = "train",
                       batch_size: int = 128,
                       num_workers: int = 8
                       ) -> torch.utils.data.DataLoader:
    """
    Create a data loader.
    
    Args:
        dataset_path (str): Path to the dataset.
        split (str): Split of the dataset to use. Either "train" or "test".
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
    ])

    if split == 'train':
        train_data_path = "data/SatellitePatches/PA-train"
        train_metadata_path = "data/GLC25_PA_metadata_train.csv"
        train_metadata = pd.read_csv(train_metadata_path)

        train_dataset = TrainDataset(train_data_path,
                                     train_metadata,
                                     transform=transform)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return train_loader
