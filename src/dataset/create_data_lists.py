"""
Module for creating data lists for the GLC25 dataset.
"""

import os
import json

def save_image_list(labels, image_paths, file_path, split: str = "train"):
    split = split.lower()
    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split. Must be 'train', 'val', or 'test'.")

    label_file = os.path.join(file_path, f"{split}_labels.json")
    image_file = os.path.join(file_path, f"{split}_images.json")

    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    with open(image_file, "w", encoding="utf-8") as f:
        json.dump(image_paths, f)