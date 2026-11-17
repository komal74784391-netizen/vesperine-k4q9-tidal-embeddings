"""Tidal position-aware token embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TidalEmbedding(nn.Module):
    """Token embedding whose positional signal decays with a learned tide rate."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_length: int = 131072):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim)
        self.max_length = max_length
        self.tide_rate = nn.Parameter(torch.tensor(1.0))
        half = hidden_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq)

    def _positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        spread = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return torch.cat([spread.sin(), spread.cos()], dim=-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        token = self.token(input_ids)
        pos = self._positions(seq_len, input_ids.device).unsqueeze(0)
        gate = torch.exp(-self.tide_rate.abs() * torch.arange(seq_len, device=input_ids.device) / self.max_length)
        gate = gate.view(1, seq_len, 1)
        return token * (1.0 + gate * pos)
