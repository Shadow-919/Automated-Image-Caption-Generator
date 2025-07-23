"""
datasets.py

Dataset preparation and loading utilities for Image Captioning project.
This includes:
- Image preprocessing and caching.
- Dataset class compatible with PyTorch DataLoader.
- Support for training, validation, and test splits.
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer
from tqdm import tqdm

from utils import transform  # Custom transform defined in utils.py


def create_image_inputs(annotation_json_path, image_dir, transform):
    """
    Preprocess and cache images as .pt files for faster loading during training.

    Args:
        annotation_json_path (str): Path to JSON file with image and caption data.
        image_dir (str): Root image directory containing subfolders like train2014/.
        transform (torchvision.transforms): Transformations to apply to images.
    """
    annotation = json.load(open(annotation_json_path, "r"))
    bar = tqdm(annotation["images"], desc="Saving preprocessed images")
    for image_info in bar:
        image_path = os.path.join(image_dir, image_info["filepath"], image_info["filename"])
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        torch.save(image, image_path.replace(".jpg", ".pt"))


class ImageCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset class for image-caption pairs.
    Supports preprocessing, loading and tokenizing captions, and training/validation/test splits.
    """

    def __init__(self, annotation_json_path, image_dir, tokenizer, max_seq_len=256, transform=None, phase="train"):
        """
        Args:
            annotation_json_path (str): Path to JSON with captions.
            image_dir (str): Directory where image folders (train2014, val2014) exist.
            tokenizer (transformers.BertTokenizer): Tokenizer for caption encoding.
            max_seq_len (int): Maximum sequence length for captions.
            transform (callable, optional): Transform to be applied on a sample image.
            phase (str): One of ['train', 'val', 'test'].
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.annotation_json_path = annotation_json_path
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.df = self._create_inputs()

    def _create_inputs(self):
        """
        Parse the annotation JSON and return a DataFrame with image-caption pairs.
        """
        df = []
        data = json.load(open(self.annotation_json_path, "r"))
        for image in data["images"]:
            image_path = os.path.join(self.image_dir, image["filepath"], image["filename"])
            captions = [" ".join(c["tokens"]) for c in image["sentences"]]
            for caption in captions:
                row = {
                    "image_id": image["cocoid"],
                    "image_path": image_path,
                    "caption": caption,
                    "all_captions": captions + [""] * (10 - len(captions))  # pad to length 10
                }
                if self.phase == "train" and image["split"] in {"train", "restval"}:
                    df.append(row)
                elif self.phase == "val" and image["split"] in {"val"}:
                    df.append(row)
                elif self.phase == "test" and image["split"] in {"test"}:
                    df.append(row)
        return pd.DataFrame(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns:
            dict: A dictionary with image, caption tokens, and additional metadata.
        """
        image_path = self.df.iloc[index]["image_path"]
        image_tensor_path = image_path.replace(".jpg", ".pt")

        if os.path.exists(image_tensor_path):
            image = torch.load(image_tensor_path)
        else:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            torch.save(image, image_tensor_path)

        caption_text = self.df.loc[index, "caption"]
        caption_tokens = self.tokenizer(
            caption_text, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"][0]

        all_captions_text = self.df.loc[index, "all_captions"]
        all_captions_tokens = self.tokenizer(
            all_captions_text, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"]

        return {
            "image_id": self.df.loc[index, "image_id"],
            "image_path": image_path,
            "image": image,
            "caption_seq": caption_text,
            "caption": caption_tokens,
            "all_captions_seq": all_captions_text,
            "all_captions": all_captions_tokens
        }


# Debug/Test script for standalone testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the ImageCaptionDataset")
    
    # Model/tokenizer arguments
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased", help="BERT tokenizer name")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128, help="Max sequence length for tokenized caption")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for DataLoader (unused here)")

    # Dataset paths
    parser.add_argument("--image_dir", "-id", type=str, default="../coco/", help="Path to image directory (with train2014, val2014 etc.)")
    parser.add_argument("--annotation_json_path", "-ajp", type=str, default="../coco/annotations/dataset_coco.json", help="Path to dataset JSON")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # Instantiate dataset
    dataset = ImageCaptionDataset(
        annotation_json_path=args.annotation_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform,
        phase="train"
    )

    print("Sample output:\n", dataset[0])

    # Preprocess and save all images
    create_image_inputs(args.annotation_json_path, args.image_dir, transform)
