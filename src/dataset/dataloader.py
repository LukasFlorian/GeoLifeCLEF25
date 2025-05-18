"""
A module containing the dataloader for the GeoLifeCLEF25 (GLC25) dataset.
"""

from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class GLC25Dataset(Dataset):
    """
    GLC25 Dataset class for loading the GeoLifeCLEF25 dataset.
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        