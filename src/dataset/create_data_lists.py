"""
Module for creating data lists for the GLC25 dataset.
"""

import os
import json
import csv

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

def parse_annotation_glc25(annotation_file_path: str, image_dir: str):
    with open(annotation_file_path, mode = "r") as f:
        reader = csv.DictReader(f)
        labels = []
        image_paths = []
        for row in reader:
            survey_id = row["survey_id"]
            if len(survey_id) == 1 or len(survey_id) == 2:
                image_path = os.path.join(image_dir, survey_id, survey_id, f"{survey_id}.tiff")
            elif len(survey_id) == 3:
                image_path = os.path.join(image_dir, survey_id[-2:], survey_id[0], f"{survey_id}.tiff")
            else:
                image_path = os.path.join(image_dir, survey_id[-2:], survey_id[-4:-2], f"{survey_id}.tiff")
            image_paths.append(image_path)
            labels.append([int(x) for x in row["labels"].split(",")]
    return labels, image_paths