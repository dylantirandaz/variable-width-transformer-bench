"""Byte-level data helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch


DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "tiny_corpus.txt"
VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(token_ids: list[int]) -> str:
    return bytes(int(i) % 256 for i in token_ids).decode("utf-8", errors="replace")


def load_bytes(path: Optional[str] = None) -> torch.Tensor:
    source = Path(path) if path else DEFAULT_CORPUS
    data = source.read_bytes()
    if len(data) < 128:
        raise ValueError(f"{source} is too small for a language-model benchmark")
    return torch.tensor(list(data), dtype=torch.long)


def train_val_split(data: torch.Tensor, val_fraction: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    split = int(len(data) * (1.0 - val_fraction))
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(
            f"data length {len(data)} must be greater than block_size + 1 ({block_size + 1})"
        )
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
