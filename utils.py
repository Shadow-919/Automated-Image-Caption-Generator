# utils.py

import torch
import matplotlib.pyplot as plt
import os
import json
from torchvision import transforms
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# ----------------------------------------------------------
# Image transformation for preprocessing input images
# ----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------
# Function: visualize_log
# Description: Saves training/validation loss and BLEU-4 plots
# ----------------------------------------------------------
def visualize_log(log, log_visualize_dir):
    # Plot loss per epoch
    plt.figure()
    plt.plot(log["train_loss"], label="train")
    plt.plot(log["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig(os.path.join(log_visualize_dir, "loss_epoch.png"))

    # Plot BLEU-4 per epoch
    plt.figure()
    plt.plot(log["train_bleu4"], label="train")
    plt.plot(log["val_bleu4"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU-4")
    plt.legend()
    plt.title("BLEU-4 per epoch")
    plt.savefig(os.path.join(log_visualize_dir, "bleu4_epoch.png"))

    # Plot loss per batch
    plt.figure()
    train_loss_batch = []
    for loss in log["train_loss_batch"]:
        train_loss_batch += loss
    plt.plot(train_loss_batch, label="train")

    val_loss_batch = []
    for loss in log["val_loss_batch"]:
        val_loss_batch += loss
    plt.plot(val_loss_batch, label="val")

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per batch")
    plt.savefig(os.path.join(log_visualize_dir, "loss_batch.png"))

# ----------------------------------------------------------
# Function: metric_scores
# Description: Evaluate predictions using COCOEvalCap
# ----------------------------------------------------------
def metric_scores(annotation_path, prediction_path):
    # Format of results JSON:
    # {"image_id": 1, "caption": "a caption"}

    results = {}
    coco = COCO(annotation_path)
    coco_result = coco.loadRes(prediction_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
        results[metric] = score

    return results

# ----------------------------------------------------------
# Function: filter_coco_by_split
# Description: Filters COCO annotations using provided split labels
#              (e.g., "train", "val", or "test") from a JSON file.
# ----------------------------------------------------------
def filter_coco_by_split(split_json_path, annotation_path, phase="test"):
    """
    Filters a COCO-format annotation file based on a specified split.

    Args:
        split_json_path (str): Path to a JSON file containing image split info.
        annotation_path (str): Path to the original COCO-style annotation file.
        phase (str): One of {"train", "val", "test"}.

    Returns:
        dict: Filtered COCO-style dictionary.
    """
    phase_set = {"train", "restval"} if phase == "train" else {phase}

    coco = json.load(open(annotation_path))
    split_info = json.load(open(split_json_path))

    filtered_ids = set([x["cocoid"] for x in split_info["images"] if x["split"] in phase_set])
    coco["images"] = [x for x in coco["images"] if x["id"] in filtered_ids]
    coco["annotations"] = [x for x in coco["annotations"] if x["image_id"] in filtered_ids]

    return coco
