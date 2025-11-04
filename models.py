"""
models.py
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable

class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
    def forward(self, x): return self.norm(x)

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i]   = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        return self.dropout(x + pe)

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        return torch.matmul(attn, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)
        q = self.q(query).view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.k(key).view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.v(value).view(b, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        scores = SelfAttention(self.dropout.p)(q, k, v, mask)
        out = scores.transpose(1, 2).contiguous().view(b, -1, self.embedding_dim)
        return self.out(out)

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(embedding_dim, ff_dim),
                                nn.ReLU(),
                                nn.Linear(ff_dim, embedding_dim))
        self.d1 = nn.Dropout(dropout); self.d2 = nn.Dropout(dropout)
        self.n1 = Norm(embedding_dim); self.n2 = Norm(embedding_dim)
    def forward(self, x, mask=None):
        x = x + self.d1(self.attn(self.n1(x), self.n1(x), self.n1(x), mask))
        x = x + self.d2(self.ff(self.n2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, encoder_layers, num_heads, dropout=0.1, depth=5, fine_tune=True):
        super().__init__()
        self.eff = EfficientNet.from_pretrained(f"efficientnet-b{depth}")
        for p in self.eff.parameters():
            p.requires_grad = fine_tune
        self.avg_pool = nn.AdaptiveAvgPool2d((max_seq_len-1, 512))
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout)
                                     for _ in range(encoder_layers)])
        self.norm = Norm(embedding_dim)

    def forward(self, image):
        features = self.eff.extract_features(image)      # (B, C, H, W)
        features = features.permute(0, 2, 3, 1)          # (B, H, W, C)
        features = features.view(features.size(0), -1, features.size(-1))
        x = self.avg_pool(features)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.enc_attn  = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(embedding_dim, ff_dim),
                                nn.ReLU(),
                                nn.Linear(ff_dim, embedding_dim))
        self.d1 = nn.Dropout(dropout); self.d2 = nn.Dropout(dropout); self.d3 = nn.Dropout(dropout)
        self.n1 = Norm(embedding_dim); self.n2 = Norm(embedding_dim); self.n3 = Norm(embedding_dim)

    def forward(self, x, memory, target_mask):
        x = x + self.d1(self.self_attn(self.n1(x), self.n1(x), self.n1(x), target_mask))
        x = x + self.d2(self.enc_attn(self.n2(x), memory, memory))
        x = x + self.d3(self.ff(self.n3(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, decoder_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout)
                                     for _ in range(decoder_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm(embedding_dim)
        self.pos = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, target, memory, target_mask):
        x = self.embed(target)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, target_mask)
        return self.norm(x)

class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, encoder_layers, decoder_layers, num_heads, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(embedding_dim, max_seq_len, encoder_layers, num_heads, dropout)
        self.decoder = Decoder(embedding_dim, vocab_size, max_seq_len, decoder_layers, num_heads, dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, image, captions):
        enc = self.encoder(image)
        mask = self.make_mask(captions)
        dec = self.decoder(captions, enc, mask)
        return self.fc(dec)

    def make_mask(self, target_ids):
        b, L = target_ids.size()
        # subsequent mask (L,L), True where allowed
        return (1 - torch.triu(torch.ones((1, L, L), device=target_ids.device), diagonal=1)).bool()
