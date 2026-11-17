"""Long-document encoder built on top of :class:`TidalEmbedding`."""

from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import TidalEmbedding


class TidalEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, layers: int = 12, heads: int = 12):
        super().__init__()
        self.embedding = TidalEmbedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask)
        return self.projection(hidden)
