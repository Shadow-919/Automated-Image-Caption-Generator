# train.py

import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import os
import json
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()

from utils import transform, visualize_log
from datasets import ImageCaptionDataset
from models import ImageCaptionModel

# -----------------------------------
# Training and validation functions
# -----------------------------------

def train_epoch(model, train_loader, tokenizer, criterion, optim, epoch, device):
    model.train()
    total_loss, batch_bleu4 = [], []
    hypotheses, references = [], []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}")
    
    for i, batch in bar:
        image, caption, all_caps = batch["image"].to(device), batch["caption"].to(device), batch["all_captions_seq"]
        target_input = caption[:, :-1]
        target_mask = model.make_mask(target_input)
        preds = model(image, target_input)

        optim.zero_grad()
        gold = caption[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), gold)
        loss.backward()
        optim.step()

        total_loss.append(loss.item())

        # BLEU-4 scoring
        preds = F.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1).detach().cpu().numpy()
        caps = [tokenizer.decode(cap, skip_special_tokens=True) for cap in preds]
        hypo = [cap.split() for cap in caps]

        batch_size = len(hypo)
        ref = []
        for i in range(batch_size):
            ri = [all_caps[j][i].split() for j in range(len(all_caps)) if all_caps[j][i]]
            ref.append(ri)
        batch_bleu4.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
        hypotheses += hypo
        references += ref

        bar.set_postfix(loss=total_loss[-1], bleu4=batch_bleu4[-1])

    train_bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss, train_bleu4, total_loss


def validate_epoch(model, valid_loader, tokenizer, criterion, epoch, device):
    model.eval()
    total_loss, batch_bleu4 = [], []
    hypotheses, references = [], []

    with torch.no_grad():
        bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}")
        for i, batch in bar:
            image, caption, all_caps = batch["image"].to(device), batch["caption"].to(device), batch["all_captions_seq"]
            target_input = caption[:, :-1]
            target_mask = model.make_mask(target_input)
            preds = model(image, target_input)

            gold = caption[:, 1:].contiguous().view(-1)
            loss = criterion(preds.view(-1, preds.size(-1)), gold)
            total_loss.append(loss.item())

            preds = F.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=-1).detach().cpu().numpy()
            caps = [tokenizer.decode(cap, skip_special_tokens=True) for cap in preds]
            hypo = [cap.split() for cap in caps]

            batch_size = len(hypo)
            ref = []
            for i in range(batch_size):
                ri = [all_caps[j][i].split() for j in range(len(all_caps)) if all_caps[j][i]]
                ref.append(ri)
            batch_bleu4.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
            hypotheses += hypo
            references += ref

            bar.set_postfix(loss=total_loss[-1], bleu4=batch_bleu4[-1])

    val_loss = sum(total_loss) / len(total_loss)
    val_bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)
    return val_loss, val_bleu4, total_loss


def train(model, train_loader, valid_loader, optim, criterion, start_epoch, n_epochs, tokenizer, device, model_path, log_path, early_stopping=5):
    model.train()
    if start_epoch > 0:
        log = json.load(open(log_path, "r"))
        best_train_bleu4, best_val_bleu4, best_epoch = log["best_train_bleu4"], log["best_val_bleu4"], log["best_epoch"]
        print("Load model from epoch {}, and continue training.".format(best_epoch))
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        log = {"train_loss": [], "train_bleu4": [], "train_loss_batch": [],
               "val_loss": [], "val_bleu4": [], "val_loss_batch": []}
        best_train_bleu4, best_val_bleu4, best_epoch = -np.Inf, -np.Inf, 1

    count_early_stopping = 0
    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        train_loss, train_bleu4, train_loss_batch = train_epoch(
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            optim=optim,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        val_loss, val_bleu4, val_loss_batch = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            criterion=criterion,
            epoch=epoch,
            device=device
        )

        best_train_bleu4 = max(train_bleu4, best_train_bleu4)

        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print("-------- Detected improvement. Saved best model --------")
            count_early_stopping = 0
        else:
            count_early_stopping += 1
            if count_early_stopping >= early_stopping:
                print("-------- Early stopping --------")
                break

        # Log training and validation metrics
        log["train_loss"].append(train_loss)
        log["train_bleu4"].append(train_bleu4)
        log["train_loss_batch"].append(train_loss_batch)
        log["val_loss"].append(val_loss)
        log["val_bleu4"].append(val_bleu4)
        log["val_loss_batch"].append(val_loss_batch)
        log["best_train_bleu4"] = best_train_bleu4
        log["best_val_bleu4"] = best_val_bleu4
        log["best_epoch"] = best_epoch
        log["last_epoch"] = epoch + 1

        with open(log_path, "w") as f:
            json.dump(log, f)

        torch.cuda.empty_cache()

        print(f"---- Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | Train BLEU-4: {train_bleu4:.5f} | Validation BLEU-4: {val_bleu4:.5f} | Best BLEU-4: {best_val_bleu4:.5f} | Best Epoch: {best_epoch} | Time: {timedelta(seconds=int(time.time()-start_time))}")

    return log


# -----------------------------------
# Main script for training
# -----------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512)
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128)
    parser.add_argument("--encoder_layers", "-ad", type=int, default=3)
    parser.add_argument("--decoder_layers", "-nl", type=int, default=6)
    parser.add_argument("--num_heads", "-nh", type=int, default=8)
    parser.add_argument("--dropout", "-dr", type=float, default=0.1)

    # Training Parameters
    parser.add_argument("--model_path", "-md", type=str, default="./pretrained/model_image_captioning_eff_transfomer.pt")
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--n_epochs", "-ne", type=int, default=20)
    parser.add_argument("--start_epoch", "-se", type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--betas", "-bt", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--eps", "-eps", type=float, default=1e-9)
    parser.add_argument("--early_stopping", "-es", type=int, default=5)

    # Data Parameters
    parser.add_argument("--image_dir", "-id", type=str, default="../coco/")
    parser.add_argument("--split_json_path", "-sjp", type=str, default="../coco/splits/dataset_split.json")
    parser.add_argument("--val_annotation_path", "-vap", type=str, default="../coco/annotations/captions_val2014.json")
    parser.add_argument("--train_annotation_path", "-tap", type=str, default="../coco/annotations/captions_train2014.json")

    # Logging
    parser.add_argument("--log_path", "-lp", type=str, default="./images/log_training.json")
    parser.add_argument("--log_visualize_dir", "-lvd", type=str, default="./images/")

    args = parser.parse_args()

    print("------------ Training Configuration ------------")
    print(args)

    os.makedirs(args.log_visualize_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    if not os.path.exists(args.image_dir):
        print(f"Directory {args.image_dir} does not exist.")
        return

    print("------------------------------------------------")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = ImageCaptionModel(
        embedding_dim=args.embedding_dim,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps)

    train_dataset = ImageCaptionDataset(
        split_json_path=args.split_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform,
        phase="train"
    )
    valid_dataset = ImageCaptionDataset(
        split_json_path=args.split_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform,
        phase="val"
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    log = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        criterion=criterion,
        start_epoch=args.start_epoch,
        n_epochs=args.n_epochs,
        tokenizer=tokenizer,
        device=device,
        model_path=args.model_path,
        log_path=args.log_path,
        early_stopping=args.early_stopping
    )

    print(f"Training finished in: {timedelta(seconds=int(time.time() - time.time()))}")
    print(f"Best Train BLEU-4: {log['best_train_bleu4']:.5f}, Best Val BLEU-4: {log['best_val_bleu4']:.5f}")
    print(f"Best Epoch: {log['best_epoch']}")

    visualize_log(log, args.log_visualize_dir)


if __name__ == "__main__":
    main()
