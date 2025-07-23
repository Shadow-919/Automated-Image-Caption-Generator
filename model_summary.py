"""
model__summary.py

This script instantiates and summarizes a Transformer-based Image Captioning model
using EfficientNet-B5 as the encoder and a Transformer decoder. It prints the model's
architecture and parameter details using `torchinfo.summary`.

Author: [Your Name]
"""

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.models import efficientnet_b5


class ImageCaptioningModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, encoder_layers, decoder_layers, num_heads, dropout):
        super(ImageCaptioningModel, self).__init__()

        # Encoder: EfficientNet-B5 without pretrained weights
        self.encoder = efficientnet_b5(weights=None)
        self.encoder.classifier = nn.Identity()  # Remove classifier for feature extraction

        # Projection: Reduce feature size from 2048 to embedding_dim (e.g., 512)
        self.enc_proj = nn.Linear(2048, embedding_dim)

        # Decoder: Transformer decoder block
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Caption embeddings and final output projection
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, images, captions):
        # Encode image to features
        features = self.encoder.features(images)
        features = features.mean([2, 3])  # Global average pooling -> [batch, 2048]
        features = self.enc_proj(features).unsqueeze(1)  # -> [batch, 1, embedding_dim]

        # Prepare caption embeddings
        embeddings = self.embedding(captions).transpose(0, 1)  # -> [seq_len, batch, embedding_dim]
        memory = features.transpose(0, 1)  # -> [1, batch, embedding_dim]

        # Decode with Transformer
        out = self.decoder(embeddings, memory)
        out = out.transpose(0, 1)  # -> [batch, seq_len, embedding_dim]
        logits = self.fc(out)  # -> [batch, seq_len, vocab_size]
        return logits


# ---------------------------------------------
# CONFIGURATION BLOCK
# ---------------------------------------------
embedding_dim = 512
vocab_size = 30522
max_seq_len = 128
encoder_layers = 6        # Not used here directly
decoder_layers = 12
num_heads = 8
dropout = 0.1

# Instantiate model
model = ImageCaptioningModel(embedding_dim, vocab_size, max_seq_len, encoder_layers, decoder_layers, num_heads, dropout)

# Load model weights
model_path = "pretrained/model_image_captioning_eff_transfomer.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)

# Dummy input data for summary
dummy_images = torch.randn(24, 3, 456, 456)  # Batch of 24
dummy_captions = torch.randint(0, vocab_size, (24, max_seq_len))  # Batch of 24

# Show model summary
summary(model, input_data=[dummy_images, dummy_captions])
