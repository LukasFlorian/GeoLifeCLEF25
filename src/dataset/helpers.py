"""
Helper functions for the dataset.
"""


import numpy as np
import os

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
