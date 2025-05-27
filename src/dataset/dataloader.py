"""
Module containing the PyTorch dataloader for the dataset.
"""


import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader


def construct_patch_path(data_path: str, survey_id: int) -> str:
    """
    Construct the patch file path based on plot_id as './CD/AB/XXXXABCD.jpeg'
    
    Args:
        data_path (str): The base path to the data.
        survey_id (str): The survey ID.
        
    Returns:
        str: The constructed patch file path.
    """
    path = data_path
    for d in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, d)

    path = os.path.join(path, f"{survey_id}.tiff")

    return path

def quantile_normalize(band: np.ndarray, low: int=2, high: int=98) -> np.ndarray:
    """
    Apply quantile normalization to a band.
    
    Args:
        band (np.ndarray): The input band to be normalized.
        low (int): The lower quantile.
        high (int): The higher quantile.
    
    Returns:
        np.ndarray: The normalized band.
    """
    # Sort the band by pixel values
    sorted_band = np.sort(band.flatten())
    # Compute the quantiles
    quantiles = np.percentile(sorted_band, np.linspace(low, high, len(sorted_band)))
    # Map the original pixel values to the quantiles
    normalized_band = np.interp(band.flatten(), sorted_band, quantiles).reshape(band.shape)

    min_val, max_val = np.min(normalized_band), np.max(normalized_band)

    # Prevent division by zero if min_val == max_val
    if max_val == min_val:
        return np.zeros_like(normalized_band, dtype=np.float32)  # Return an array of zeros

    # Perform normalization (min-max scaling)
    return ((normalized_band - min_val) / (max_val - min_val)).astype(np.float32)

class TrainDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing training data. This class inherits form PyTorch's Dataset class.
    """
    def __init__(self, data_dir, metadata, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the training images.
            metadata (pd.DataFrame): DataFrame containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data_dir = data_dir

        self.metadata = metadata
        # Drop rows with missing speciesId
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        # Convert speciesId to integer type
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)

        # Create a dictionary mapping surveyId to a list of speciesId occurring in that survey
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()

        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

        self.num_classes = 11255

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(self.num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species

        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array([quantile_normalize(band) for band in image])  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        image = self.transform(image)

        return image, label, survey_id

class TestDataset(TrainDataset):
    """
    Custom dataset class for loading and preprocessing test data. This class inherits form PyTorch's Dataset class.
    """
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]

        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array([quantile_normalize(band) for band in image])  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format

        image = self.transform(image)
        return image, survey_id
