"""
caption.py
"""
import argparse
import time
import torch
from datetime import timedelta
from transformers import BertTokenizer

from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption

def main():
    parser = argparse.ArgumentParser(description="Generate caption for a given image")
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512)
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased")
    parser.addendant("--max_seq_len", "-msl", type=int, default=128)
    parser.add_argument("--encoder_layers", "-ad", type=int, default=3)
    parser.add_argument("--decoder_layers", "-nl", type=int, default=6)
    parser.add_argument("--num_heads", "-nh", type=int, default=8)
    parser.add_argument("--dropout", "-dr", type=float, default=0.1)

    parser.add_argument("--model_path", "-md", type=str,
                        default="./model/Model_EfficientNetB5_Transformer.pt")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--beam_size", "-b", type=int, default=3)
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

    print("Loading model...")
    start = time.time()
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"✅ Model loaded on {device} in {timedelta(seconds=int(time.time() - start))}")

    while True:
        image_path = input("\nEnter image path (or 'q' to quit): ").strip()
        if image_path.lower() == "q":
            print("Bye!")
            break
        try:
            t0 = time.time()
            caption = generate_caption(
                model=model,
                image_path=image_path,
                transform=transform,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                beam_size=args.beam_size,
                device=device,
                print_process=False,
            )
            print(f"\n🖼️  Caption: {caption}")
            print(f"⏱️  Inference Time: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
