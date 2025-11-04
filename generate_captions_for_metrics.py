import argparse
import json
import os
import torch
from transformers import BertTokenizer

from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption

def main():
    parser = argparse.ArgumentParser(description="Batch caption generator for metrics")
    parser.add_argument("--images_list", type=str, required=True,
                        help="Path to a JSON list of {image_path, image_id}")
    parser.add_argument("--output_path", type=str, default="./results/predictions.json")

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--model_path", type=str,
                        default="./model/Model_EfficientNetB5_Transformer.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--beam_size", type=int, default=3)

    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    model = ImageCaptionModel(
        embedding_dim=args.embedding_dim,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    with open(args.images_list, "r") as f:
        items = json.load(f)

    preds = []
    for it in items:
        cap = generate_caption(
            model=model,
            image_path=it["image_path"],
            transform=transform,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            beam_size=args.beam_size,
            device=device,
            print_process=False,
        )
        preds.append({"image_id": it["image_id"], "caption": cap})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
