import matplotlib.pyplot as plt
import os
from torchvision import transforms

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def visualize_log(log, log_visualize_dir):
    os.makedirs(log_visualize_dir, exist_ok=True)

    # Loss per epoch
    plt.figure()
    plt.plot(log.get("train_loss", []), label="train")
    plt.plot(log.get("val_loss", []), label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss per epoch")
    plt.savefig(os.path.join(log_visualize_dir, "loss_epoch.png"))

    # BLEU-4 per epoch
    plt.figure()
    plt.plot(log.get("train_bleu4", []), label="train")
    plt.plot(log.get("val_bleu4", []), label="val")
    plt.xlabel("Epoch"); plt.ylabel("BLEU-4"); plt.legend()
    plt.title("BLEU-4 per epoch")
    plt.savefig(os.path.join(log_visualize_dir, "bleu4_epoch.png"))

    # Loss per batch
    plt.figure()
    train_loss_batch = [v for sub in log.get("train_loss_batch", []) for v in sub]
    val_loss_batch   = [v for sub in log.get("val_loss_batch", [])   for v in sub]
    if train_loss_batch:
        plt.plot(train_loss_batch, label="train")
    if val_loss_batch:
        plt.plot(val_loss_batch, label="val")
    plt.xlabel("Batch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss per batch")
    plt.savefig(os.path.join(log_visualize_dir, "loss_batch.png"))
