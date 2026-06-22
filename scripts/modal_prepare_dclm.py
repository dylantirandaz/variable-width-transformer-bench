#!/usr/bin/env python3
"""Prepare the DCLM cl100k memmap inside a Modal volume.

The paper-scale token budget counts input tokens consumed by training. A
next-token language-modeling batch also needs one terminal target token after
the final input token, so the stored memmap contains ``training_tokens + 1``
tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


APP_DIR = "/root/vwt-bench"
DATA_DIR = "/data"
DENSE_TOTAL_TOKENS = {
    "dense_200m": 10_003_415_040,
    "dense_500m": 25_001_328_384,
    "dense_1b": 49_999_724_544,
    "dense_2b": 104_857_600_000,
}

app = modal.App("variable-width-transformer-dclm-prep")
data_volume = modal.Volume.from_name("vwt-paper-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "tiktoken>=0.7", "datasets>=2.20", "zstandard>=0.22")
    .add_local_dir("src", f"{APP_DIR}/src")
    .add_local_dir("scripts", f"{APP_DIR}/scripts")
)


@app.function(
    image=image,
    cpu=16.0,
    memory=32_768,
    timeout=24 * 60 * 60,
    volumes={DATA_DIR: data_volume},
)
def prepare_dclm(
    scale: str = "dense_200m",
    output: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    dataset: str = "mlfoundations/dclm-baseline-1.0",
    text_key: str = "text",
    overwrite: bool = False,
) -> dict:
    if scale not in DENSE_TOTAL_TOKENS:
        allowed = ", ".join(sorted(DENSE_TOTAL_TOKENS))
        raise ValueError(f"unknown scale {scale!r}; expected one of: {allowed}")
    training_tokens = DENSE_TOTAL_TOKENS[scale]
    target_tokens = training_tokens + 1
    output_path = Path(output)
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tokens", 0) >= target_tokens:
            return {
                "status": "exists",
                "output": str(output_path),
                "manifest": str(manifest_path),
                "tokens": manifest["tokens"],
                "training_tokens": training_tokens,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/prepare_cl100k_dataset.py",
        "--hf-dataset",
        dataset,
        "--hf-split",
        "train",
        "--text-key",
        text_key,
        "--output",
        str(output_path),
        "--manifest",
        str(manifest_path),
        "--max-tokens",
        str(target_tokens),
        "--log-interval",
        "5000",
    ]
    run(cmd)
    data_volume.commit()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "status": "created",
        "output": str(output_path),
        "manifest": str(manifest_path),
        "tokens": manifest["tokens"],
        "training_tokens": training_tokens,
        "documents": manifest["documents"],
    }


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_DIR}/src"
    subprocess.run(
        cmd,
        cwd=APP_DIR,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main(
    scale: str = "dense_200m",
    output: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    dataset: str = "mlfoundations/dclm-baseline-1.0",
    overwrite: bool = False,
) -> None:
    result = prepare_dclm.remote(scale=scale, output=output, dataset=dataset, overwrite=overwrite)
    print(json.dumps(result, indent=2))
