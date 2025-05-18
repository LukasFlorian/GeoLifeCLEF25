"""
Module for creating data lists for the GLC25 dataset.
"""

import os
import json

def save_image_list(labels: list[list[int]], image_paths: list[str], file_path, split: str = "train"):
    """
    Saves the image list and corresponding labels to JSON files.

    Args:
        labels (_type_): _description_
        image_paths (_type_): _description_
        file_path (_type_): _description_
        split (str, optional): _description_. Defaults to "train".

    Raises:
        ValueError: _description_
    """
    split = split.lower()
    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split. Must be 'train', 'val', or 'test'.")

    label_file = os.path.join(file_path, f"{split}_labels.json")
    image_file = os.path.join(file_path, f"{split}_images.json")

    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    with open(image_file, "w", encoding="utf-8") as f:
        json.dump(image_paths, f)
    print(f"Saved {split} data lists containing {len(labels)} data points to {file_path}")

