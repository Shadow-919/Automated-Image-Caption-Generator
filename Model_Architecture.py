# model_architecture.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5

# Define the Transformer-based Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, encoder_layers, decoder_layers, num_heads, dropout):
        super(ImageCaptioningModel, self).__init__()

        # EfficientNetB5 as encoder, remove final classification head
        self.encoder = efficientnet_b5(weights=None)
        self.encoder.classifier = nn.Identity()
        
        # Projection from encoder output (2048) to embedding dimension
        self.enc_proj = nn.Linear(2048, embedding_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Word embedding and output vocabulary projection
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, images, captions):
        # Extract encoder features
        features = self.encoder.features(images)
        features = features.mean([2, 3])  # Global Average Pooling
        features = self.enc_proj(features).unsqueeze(1)  # Add sequence dimension

        # Embed captions
        embeddings = self.embedding(captions).transpose(0, 1)
        memory = features.transpose(0, 1)

        # Transformer decoding
        out = self.decoder(embeddings, memory).transpose(0, 1)

        # Final vocabulary projection
        logits = self.fc(out)
        return logits

# Model configuration
embedding_dim = 512
vocab_size = 30522
max_seq_len = 128
encoder_layers = 6           # Placeholder (not used)
decoder_layers = 12
num_heads = 8
dropout = 0.1

# Initialize model
model = ImageCaptioningModel(
    embedding_dim, vocab_size, max_seq_len,
    encoder_layers, decoder_layers, num_heads, dropout
)

# Load trained weights
model_path = "pretrained/model_image_captioning_eff_transfomer.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)

# Save architecture to file
output_path = "model_architecture.txt"
with open(output_path, "w") as f:
    f.write("----- START OF MODEL ARCHITECTURE -----\n\n")
    f.write(str(model))
    f.write("\n\n----- END OF MODEL ARCHITECTURE -----")

print(f"✅ Model architecture saved to {output_path}")
