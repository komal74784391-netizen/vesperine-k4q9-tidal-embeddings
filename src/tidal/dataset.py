"""Dataset utilities for long documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class LongDocumentDataset(Dataset):
    samples: list[dict]
    vocab_size: int = 50257
    max_length: int = 131072

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        ids = torch.tensor(sample["input_ids"][: self.max_length], dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[: ids.size(0)] = True
        labels = ids.clone()
        return {"input_ids": ids, "attention_mask": ~mask, "labels": labels}

    @classmethod
    def from_config(cls, config_path: str) -> "LongDocumentDataset":
        cfg = json.loads(Path(config_path).read_text())
        return cls(samples=cfg["samples"], vocab_size=cfg.get("vocab_size", 50257))
