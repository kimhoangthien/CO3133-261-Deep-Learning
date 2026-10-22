"""Transformer encoder classifier for Fashion-MNIST, owned by Antoine.

Flow: image -> tokens -> projection -> positional encoding -> encoder
-> aggregation -> classifier -> logits.
Input: [B, 1, 28, 28]. Output: [B, num_classes] raw logits.

Tokens are image rows or square patches, so the same encoder can be read
as a sequence model over rows (the representation the recurrent model
uses) or as a patch model in the ViT sense. Positional encoding is
sinusoidal by default, which costs no parameters, or a learned table.
"""

import math
import torch
from torch import nn


def sinusoidal_encoding(length, d_model):
    """PE(pos, 2i) = sin(pos / 10000 ** (2i / d)), cos on odd channels."""
    position = torch.arange(length).unsqueeze(1).float()
    divisor = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class TransformerClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, tokens="rows", patch_size=7,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, positional="sinusoidal", pooling="cls"):
        super().__init__()
        channels, height, width = input_shape
        if channels != 1:
            raise ValueError("input_shape must be single-channel for this classifier")
        if d_model % nhead or d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        if tokens == "rows":
            self.sequence_length, token_size = height, width
        elif tokens == "patches":
            if height % patch_size or width % patch_size:
                raise ValueError("patch_size must divide both image dimensions")
            self.sequence_length = (height // patch_size) * (width // patch_size)
            token_size = patch_size * patch_size
        else:
            raise ValueError("tokens must be 'rows' or 'patches'")
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.tokens, self.patch_size, self.pooling = tokens, patch_size, pooling

        self.projection = nn.Linear(token_size, d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        positions = self.sequence_length + (pooling == "cls")
        if positional == "sinusoidal":
            self.register_buffer("position", sinusoidal_encoding(positions, d_model).unsqueeze(0))
        elif positional == "learned":
            self.position = nn.Parameter(torch.zeros(1, positions, d_model))
        else:
            raise ValueError("positional must be 'sinusoidal' or 'learned'")
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def tokenize(self, images):
        """[B, 1, H, W] -> [B, sequence_length, token_size]."""
        if self.tokens == "rows":
            return images.squeeze(1)
        patches = images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        return patches.reshape(images.size(0), self.sequence_length, -1)

    def forward(self, inputs):
        if inputs.dim() != 4 or inputs.size(1) != 1:
            raise ValueError(f"expected [B, 1, H, W] images, got {tuple(inputs.shape)}")
        sequence = self.projection(self.tokenize(inputs))
        if self.class_token is not None:
            sequence = torch.cat([self.class_token.expand(sequence.size(0), -1, -1), sequence], 1)
        # Attention reads [B, tokens, d_model] and returns the same shape.
        encoded = self.encoder(self.dropout(sequence + self.position))
        pooled = encoded[:, 0] if self.pooling == "cls" else encoded.mean(1)
        return self.head(pooled)
