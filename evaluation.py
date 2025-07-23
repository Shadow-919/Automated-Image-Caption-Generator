# evaluation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
import PIL
from tqdm import tqdm
from datetime import timedelta
from transformers import BertTokenizer
from utils import transform, metric_scores, filter_coco_by_split
from models import ImageCaptionModel
from datasets import ImageCaptionDataset


def generate_caption(model, image_path, transform, tokenizer, max_seq_len=256, beam_size=3, device=torch.device("cpu"), print_process=False):
    """
    Generates a caption for a single image using beam search.
    """
    image = PIL.Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(image)
        beams = [([tokenizer.cls_token_id], 0)]
        completed = []

        for step in range(max_seq_len):
            new_beams = []
            for seq, score in beams:
                input_token = torch.tensor([seq]).to(device)
                target_mask = model.make_mask(input_token).to(device)
                pred = model.decoder(input_token, encoder_output, target_mask)
                pred = F.softmax(model.fc(pred), dim=-1)
                pred = pred[:, -1, :].view(-1)

                top_k_scores, top_k_tokens = pred.topk(beam_size)
                for i in range(beam_size):
                    new_seq = seq + [top_k_tokens[i].item()]
                    new_score = score + top_k_scores[i].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            for beam in beams[:]:
                if beam[0][-1] == tokenizer.sep_token_id:
                    completed.append(beam)
                    beams.remove(beam)
                    beam_size -= 1

            if print_process:
                print(f"Step {step + 1}/{max_seq_len}")
                print(f"Remaining Beams: {[tokenizer.decode(b[0]) for b in beams]}")
                print(f"Completed: {[tokenizer.decode(b[0]) for b in completed]}")
                print("-" * 100)

            if beam_size == 0:
                break

        completed = completed or beams  # fallback in case no <SEP> reached
        completed.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        return tokenizer.decode(best_seq, skip_special_tokens=True)


def evaluate():
    """
    Evaluates the model on the COCO test split and calculates metrics.
    """
    import argparse
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512)
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128)
    parser.add_argument("--encoder_layers", "-ad", type=int, default=3)
    parser.add_argument("--decoder_layers", "-nl", type=int, default=6)
    parser.add_argument("--num_heads", "-nh", type=int, default=8)
    parser.add_argument("--dropout", "-dr", type=float, default=0.1)

    # File paths
    parser.add_argument("--model_path", "-md", type=str, default="./model/Model_EfficientNetB5_Transformer.pt")
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--image_dir", "-id", type=str, default="../coco/")
    parser.add_argument("--data_json", "-dj", type=str, default="../coco/dataset_coco.json")
    parser.add_argument("--val_annotation_path", "-vap", type=str, default="../coco/annotations/captions_val2014.json")
    parser.add_argument("--train_annotation_path", "-tap", type=str, default="../coco/annotations/captions_train2014.json")
    parser.add_argument("--output_dir", "-o", type=str, default="./results/")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Initialize model
    model_configs = {
        "embedding_dim": args.embedding_dim,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_seq_len,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }

    start_time = time.time()
    model = ImageCaptionModel(**model_configs)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()
    load_time = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"✅ Loaded model on {device} in {load_time}")

    # Load image paths from dataset split
    with open(args.data_json, "r") as f:
        split_data = json.load(f)

    image_paths = [
        os.path.join(args.image_dir, img["filepath"], img["filename"])
        for img in split_data["images"] if img["split"] == "test"
    ]
    image_ids = [img["cocoid"] for img in split_data["images"] if img["split"] == "test"]

    # Convert dataset split to COCO format
    coco_ann = filter_coco_by_split(args.data_json, args.val_annotation_path)
    ann_path = os.path.join(args.output_dir, "coco_annotation_test.json")
    with open(ann_path, "w") as f:
        json.dump(coco_ann, f)

    # Evaluate with beam search
    beam_sizes = [3]
    scores = {}

    for beam_size in beam_sizes:
        predictions = []
        for image_path, image_id in tqdm(zip(image_paths, image_ids), total=len(image_paths)):
            caption = generate_caption(
                model, image_path, transform, tokenizer,
                beam_size=beam_size, device=device
            )
            predictions.append({"image_id": image_id, "caption": caption})

        # Save predictions
        pred_path = os.path.join(args.output_dir, f"prediction_beam_size_{beam_size}.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)

        # Calculate scores
        score = metric_scores(annotation_path=ann_path, prediction_path=pred_path)
        scores[f"beam{beam_size}"] = score

    # Save evaluation scores
    scores_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f)

    # Report results
    print("🎯 Evaluation Complete")
    print(f"Scores saved to {scores_path}")
    for beam_size in beam_sizes:
        print(f"\n📘 Beam size: {beam_size}")
        print(f"📄 Prediction saved to: prediction_beam_size_{beam_size}.json")
        for metric, val in scores[f"beam{beam_size}"].items():
            print(f"  • {metric}: {val:.4f}")


if __name__ == "__main__":
    evaluate()
