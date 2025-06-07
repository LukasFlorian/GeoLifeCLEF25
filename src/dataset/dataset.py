"""
Custom PyTorch Dataset subclass for the GLC25 dataset.
"""


import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from .helpers import quantile_normalize, construct_patch_path


class TrainDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing training data. This class inherits form PyTorch's Dataset class.
    """
    def __init__(self,
                 data_dir: str,
                 metadata: pd.DataFrame,
                 transform=None,
                 grid_length: float | None = None,
                 pseudo_label_generator: torch.nn.Module | None = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the training images.
            metadata (pd.DataFrame): DataFrame containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        if grid_length and pseudo_label_generator:
            raise ValueError("Cannot use both grid_length and pseudo_label_generator.")
        
        self.num_classes = 11255
        self.transform = transform
        self.data_dir = data_dir

        self.metadata = metadata
        # Drop rows with missing speciesId
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        # Convert speciesId to integer type
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)

        if grid_length:
            self.label_dict = self.box_label_union(grid_length)

        elif pseudo_label_generator:
            self.label_dict = self.pseudo_labels(pseudo_label_generator)

        else:
            self.label_dict = self.plain_labels()

        self.drop_duplicates()
    
    def drop_duplicates(self):
        """
        Drop duplicate rows from the metadata DataFrame based on the 'surveyId' column.
        """
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

    def plain_labels(self) -> dict[int, list[int]]:
        """
        Return the plain labels for a given survey.
        """
        return self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()

    def box_label_union(self,
                        grid_length: float = 0.01
                        ) -> dict[int, list[int]]:
        
        # Generate the plain label dict that maps surveyId to speciesId        
        plain_labels = self.plain_labels()
        
        # Drop duplicate rows from the metadata DataFrame based on the 'surveyId' column.
        self.drop_duplicates()

        # Get the minimum latitute and longitude from the metadata DataFrame
        min_lat = self.metadata['lat'].min()
        min_lon = self.metadata['lon'].min()

        # Get the row and col index in the grid for each surveyId
        row = self.metadata['lat'].apply(lambda x: int((x - min_lat) / grid_length))
        col = self.metadata['lon'].apply(lambda x: int((x - min_lon) / grid_length))

        # Generate the box id for each surveyId
        self.metadata["box"] = self.metadata.apply(lambda x: f"{row.loc[x.name]}_{col.loc[x.name]}", axis=1)

        # Mapping from box to surveys in the box
        box_to_surveys_map = self.metadata.groupby('box')['surveyId'].apply(list)

        # Mapping from box to species (label union) in the box
        box_to_species_map = box_to_surveys_map.apply(lambda x: list(set([speciesId for survey in x for speciesId in plain_labels[survey]])))

        # Assignment of box labels to surveys
        labels = self.metadata['box'].apply(lambda x: box_to_species_map[x]).to_dict()

        return labels


    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Return one sample of data.
        """

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
