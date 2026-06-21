from pathlib import Path

import torch

from vwt_bench.data import _unwrap_soft_line_breaks, load_bytes


def test_unwrap_soft_line_breaks_preserves_paragraphs() -> None:
    text = "First line\ncontinues here.\n\nSecond paragraph\ncontinues too.\n"

    assert _unwrap_soft_line_breaks(text) == "First line continues here.\n\nSecond paragraph continues too.\n"


def test_load_bytes_preserves_custom_file_bytes(tmp_path: Path) -> None:
    custom = tmp_path / "custom.txt"
    payload = ("alpha\nbeta\n" * 20).encode("utf-8")
    custom.write_bytes(payload)

    loaded = load_bytes(str(custom))

    assert torch.equal(loaded, torch.tensor(list(payload), dtype=torch.long))
