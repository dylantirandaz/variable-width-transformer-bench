"""Memmapped token data for large pretraining runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch


TokenDType = Literal["uint16", "uint32", "int64"]

TOKEN_DTYPES: dict[str, np.dtype] = {
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "int64": np.dtype("int64"),
}


class MemmapTokenDataset:
    """Read contiguous token sequences from a one-dimensional token memmap."""

    def __init__(self, path: str | Path, dtype: TokenDType = "uint32") -> None:
        if dtype not in TOKEN_DTYPES:
            allowed = ", ".join(TOKEN_DTYPES)
            raise ValueError(f"unknown dtype {dtype!r}; expected one of: {allowed}")
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.dtype = dtype
        self.tokens = np.memmap(self.path, mode="r", dtype=TOKEN_DTYPES[dtype])
        if self.tokens.ndim != 1:
            raise ValueError("token memmap must be one-dimensional")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def available_sequences(self, sequence_length: int) -> int:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return max((len(self) - 1) // sequence_length, 0)

    def require_tokens(self, tokens: int) -> None:
        if len(self) < tokens + 1:
            raise ValueError(
                f"{self.path} contains {len(self):,} tokens, but this run needs at least {tokens + 1:,}"
            )

    def sequential_batch(
        self,
        start_sequence: int,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_sequence < 0:
            raise ValueError("start_sequence must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        end_sequence = start_sequence + batch_size
        last_target = end_sequence * sequence_length + 1
        if last_target > len(self):
            raise ValueError(
                f"batch ending at token {last_target:,} exceeds dataset length {len(self):,}"
            )

        starts = (start_sequence + np.arange(batch_size, dtype=np.int64)) * sequence_length
        x_np = np.stack([self.tokens[start : start + sequence_length] for start in starts])
        y_np = np.stack([self.tokens[start + 1 : start + sequence_length + 1] for start in starts])
        x = torch.from_numpy(np.asarray(x_np, dtype=np.int64)).to(device, non_blocking=True)
        y = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).to(device, non_blocking=True)
        return x, y
