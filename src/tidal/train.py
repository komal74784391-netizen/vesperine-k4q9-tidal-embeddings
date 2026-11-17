"""Training entry point for the Tidal Embeddings experiments."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .encoder import TidalEncoder
from .dataset import LongDocumentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    dataset = LongDocumentDataset.from_config(args.config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TidalEncoder(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={running / (step + 1):.4f}")
