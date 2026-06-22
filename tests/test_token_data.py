from pathlib import Path

import numpy as np
import torch

from vwt_bench.token_data import MemmapTokenDataset


def test_memmap_token_dataset_reads_sequential_batches(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint32).tofile(path)
    dataset = MemmapTokenDataset(path)

    x, y = dataset.sequential_batch(
        start_sequence=2,
        batch_size=3,
        sequence_length=4,
        device=torch.device("cpu"),
    )

    assert x.tolist() == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ]
    assert y.tolist() == [
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    assert dataset.available_sequences(4) == 7


def test_memmap_token_dataset_rejects_short_data(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(8, dtype=np.uint32).tofile(path)
    dataset = MemmapTokenDataset(path)

    try:
        dataset.require_tokens(8)
    except ValueError as exc:
        assert "needs at least" in str(exc)
    else:
        raise AssertionError("expected short token dataset to fail")
