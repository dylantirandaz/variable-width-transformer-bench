#!/usr/bin/env python3
"""Append the terminal cl100k target token needed by paper-scale training."""

from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np


DATA_DIR = "/data"
CL100K_EOS_TOKEN = 100_261
DENSE_TOTAL_TOKENS = {
    "dense_200m": 10_003_415_040,
    "dense_500m": 25_001_328_384,
    "dense_1b": 49_999_724_544,
    "dense_2b": 104_857_600_000,
}

app = modal.App("variable-width-transformer-data-maintenance")
data_volume = modal.Volume.from_name("vwt-paper-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26")


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
    volumes={DATA_DIR: data_volume},
)
def append_terminal_token(
    scale: str = "dense_200m",
    output: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    token: int = CL100K_EOS_TOKEN,
) -> dict:
    if scale not in DENSE_TOTAL_TOKENS:
        allowed = ", ".join(sorted(DENSE_TOTAL_TOKENS))
        raise ValueError(f"unknown scale {scale!r}; expected one of: {allowed}")

    training_tokens = DENSE_TOTAL_TOKENS[scale]
    expected_storage_tokens = training_tokens + 1
    output_path = Path(output)
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    if not output_path.exists():
        raise FileNotFoundError(output_path)

    itemsize = np.dtype("uint32").itemsize
    size_bytes = output_path.stat().st_size
    if size_bytes % itemsize != 0:
        raise ValueError(f"{output_path} has non-uint32 byte length {size_bytes}")

    current_tokens = size_bytes // itemsize
    if current_tokens == expected_storage_tokens:
        return {
            "status": "exists",
            "output": str(output_path),
            "tokens": current_tokens,
            "training_tokens": training_tokens,
        }
    if current_tokens != training_tokens:
        raise ValueError(
            f"{output_path} has {current_tokens:,} tokens; expected "
            f"{training_tokens:,} or {expected_storage_tokens:,}"
        )

    with output_path.open("ab") as writer:
        np.asarray([token], dtype=np.uint32).tofile(writer)

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "output": str(output_path),
            "dtype": "uint32",
            "tokens": expected_storage_tokens,
            "training_tokens": training_tokens,
            "terminal_target_token": token,
            "terminal_target_token_added": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    data_volume.commit()
    return {
        "status": "appended",
        "output": str(output_path),
        "manifest": str(manifest_path),
        "tokens": expected_storage_tokens,
        "training_tokens": training_tokens,
        "terminal_target_token": token,
    }


@app.local_entrypoint()
def main(
    scale: str = "dense_200m",
    output: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    token: int = CL100K_EOS_TOKEN,
) -> None:
    result = append_terminal_token.remote(scale=scale, output=output, token=token)
    print(json.dumps(result, indent=2))
