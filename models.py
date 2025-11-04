"""
models.py
EfficientNet + Transformer for Image Captioning
FP16 safe inference version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
import math
import numpy as np


# ---------- Embedding ----------
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# ---------- LayerNorm Wrapper ----------
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# ---------- Positional Encoder (FP16 safe) ----------
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embedding_dim)))

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)

        # FP32 positional encoding then convert back to FP16
        pe = self.pe[:, :seq_len].to(x.device, dtype=torch.float32)
        x = x.float() + pe
        x = x.half()

        return self.dropout(x)


# ---------- Self Attention (FP16 safe softmax) ----------
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        dk = key.size(-1)
        scores = torch.matmul(query / np.sqrt(dk), key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax in fp32 for stability
        attn = torch.softmax(scores.float(), dim=-1).half()
        attn = self.dropout(attn)

        return torch.matmul(attn, value)


# ---------- Multi-head attention ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.self_attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        def shape(x):
            return x.view(B, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        q = shape(self.query_projection(query))
        k = shape(self.key_projection(key))
        v = shape(self.value_projection(value))

        out = self.self_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.dim_per_head)
        return self.out(out)


# ---------- Encoder ----------
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, layers, num_heads, dropout=0.1, depth=5, fine_tune=True):
        super().__init__()
        self.eff = EfficientNet.from_pretrained(f"efficientnet-b{depth}")
        self.set_fine_tune(fine_tune)
        self.avg_pool = nn.AdaptiveAvgPool2d((max_seq_len - 1, 512))
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(layers)])
        self.norm = Norm(embedding_dim)

    def forward(self, img):
        feat = self.eff.extract_features(img)
        feat = feat.permute(0, 2, 3, 1).view(feat.size(0), -1, feat.size(-1))
        x = self.avg_pool(feat)

        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def set_fine_tune(self, fine_tune):
        for p in self.eff.parameters():
            p.requires_grad = fine_tune


# ---------- Decoder ----------
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.enc_attn(x2, memory, memory))
        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_len, layers, num_heads, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos = PositionalEncoder(embedding_dim, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(layers)])
        self.norm = Norm(embedding_dim)

    def forward(self, tgt, memory, mask):
        x = self.embed(tgt)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, mask)
        return self.norm(x)


# ---------- Full Caption Model ----------
class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, encoder_layers, decoder_layers, num_heads, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(embedding_dim, max_seq_len, encoder_layers, num_heads, dropout)
        self.decoder = Decoder(embedding_dim, vocab_size, max_seq_len, decoder_layers, num_heads, dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, img, tgt):
        memory = self.encoder(img)
        mask = self.make_mask(tgt)
        out = self.decoder(tgt, memory, mask)
        return self.fc(out)

    def make_mask(self, tgt):
        L = tgt.size(1)
        mask = torch.triu(torch.ones((1, L, L), device=tgt.device), diagonal=1) == 0
        return mask
