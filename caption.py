"""
caption.py

This script allows interactive caption generation for input images using a pretrained image captioning model.
It uses beam search to produce high-quality results.

Usage:
    python caption.py --model_path ./model/Model_EfficientNetB5_Transformer.pt
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
    parser = argparse.ArgumentParser(description="Generate caption for a given image interactively")

    # Model architecture parameters
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased", help="BERT tokenizer name")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128, help="Maximum sequence length for caption generation")
    parser.add_argument("--encoder_layers", "-ad", type=int, default=3, help="Number of layers in the transformer encoder")
    parser.add_argument("--decoder_layers", "-nl", type=int, default=6, help="Number of layers in the transformer decoder")
    parser.add_argument("--num_heads", "-nh", type=int, default=8, help="Number of heads in multi-head attention")
    parser.add_argument("--dropout", "-dr", type=float, default=0.1, help="Dropout rate")

    # Model & inference parameters
    parser.add_argument("--model_path", "-md", type=str, default="./pretrained/model_image_captioning_eff_transfomer.pt", help="Path to the trained model weights")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Computation device: cpu or cuda")
    parser.add_argument("--beam_size", "-b", type=int, default=3, help="Beam size for beam search")

    args = parser.parse_args()

    # Load tokenizer and model
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    model_configs = {
        "embedding_dim": args.embedding_dim,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_seq_len,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }

    print("Loading model...")
    start_time = time.time()

    model = ImageCaptionModel(**model_configs)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    
    # Reduce memory: FP16
    model.half()
    
    model.to(device)
    model.eval()
    
    # Ensure input stays FP32 for image, convert to FP16 inside model


    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"✅ Model loaded on {device} in {elapsed_time}")

    # Interactive loop
    while True:
        image_path = input("\nEnter image path (or type 'q' to quit): ").strip()
        if image_path.lower() == "q":
            print("Exiting...")
            break
        try:
            start = time.time()
            caption = generate_caption(
                model=model,
                image_path=image_path,
                transform=transform,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                beam_size=args.beam_size,
                device=device,
                print_process=False
            )
            end = time.time()
            print(f"\n🖼️  Caption: {caption}")
            print(f"⏱️  Inference Time: {end - start:.2f} seconds")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
