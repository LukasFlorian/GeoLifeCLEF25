"""
Custom PyTorch Dataset subclass for the GLC25 dataset.
"""


import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
from .helpers import quantile_normalize, construct_patch_path


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
