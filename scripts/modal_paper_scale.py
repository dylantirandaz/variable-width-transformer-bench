#!/usr/bin/env python3
"""Launch paper-scale training on an 8xA100 Modal container.

The training data is expected in the Modal volume ``vwt-paper-data``. Keep data
preparation separate from training so a 10B-token dataset is not uploaded by
accident.

Example:
    modal run scripts/modal_paper_scale.py \
      --scale dense_200m \
      --model-kind both \
      --train-bin /data/dclm_cl100k_uint32.bin

By default the launcher keeps the two newest checkpoints per scale/model,
commits the Modal volume after each checkpoint, resumes from the latest matching
checkpoint, and skips a model kind when its final report already exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import modal


APP_DIR = "/root/vwt-bench"
DATA_DIR = "/data"
OUTPUT_DIR = "/outputs"

app = modal.App("variable-width-transformer-paper-scale")
runs_volume = modal.Volume.from_name("vwt-bench-runs", create_if_missing=True)
data_volume = modal.Volume.from_name("vwt-paper-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy>=1.26")
    .add_local_dir("src", f"{APP_DIR}/src")
    .add_local_dir("scripts", f"{APP_DIR}/scripts")
)


@app.function(
    image=image,
    gpu="A100:8",
    timeout=24 * 60 * 60,
    volumes={OUTPUT_DIR: runs_volume, DATA_DIR: data_volume},
)
def run_paper_scale(
    scale: str = "dense_200m",
    model_kind: str = "both",
    train_bin: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    checkpoint_interval: int = 1000,
    checkpoint_at_end: bool = True,
    keep_checkpoints: int = 2,
    resume: bool = True,
) -> dict:
    if model_kind not in {"constant", "variable", "both"}:
        raise ValueError("model_kind must be constant, variable, or both")
    kinds = ["constant", "variable"] if model_kind == "both" else [model_kind]
    reports = {}
    for kind in kinds:
        report_path = Path(OUTPUT_DIR) / f"{scale}_{kind}.json"
        checkpoint_dir = Path(OUTPUT_DIR) / "paper_scale_checkpoints"
        if resume and report_path.exists():
            print(f"report exists, skipping {kind}: {report_path}", flush=True)
            reports[kind] = json.loads(report_path.read_text(encoding="utf-8"))
            continue

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "scripts/train_paper_scale.py",
            "--scale",
            scale,
            "--model-kind",
            kind,
            "--train-bin",
            train_bin,
            "--report-path",
            str(report_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-interval",
            str(checkpoint_interval),
            "--keep-checkpoints",
            str(keep_checkpoints),
        ]
        if not checkpoint_at_end:
            cmd.append("--no-checkpoint-at-end")
        if resume:
            checkpoint = latest_checkpoint(checkpoint_dir, scale, kind)
            if checkpoint:
                print(f"resuming {kind} from {checkpoint}", flush=True)
                cmd.extend(["--resume", str(checkpoint)])
            else:
                print(f"no checkpoint for {kind}; starting from step 1", flush=True)
        run(cmd, commit_on_checkpoint=True)
        reports[kind] = json.loads(report_path.read_text(encoding="utf-8"))
        runs_volume.commit()
    comparison_path = None
    if set(kinds) == {"constant", "variable"}:
        comparison_path = Path(OUTPUT_DIR) / f"{scale}_comparison.md"
        run(
            [
                sys.executable,
                "scripts/compare_paper_scale.py",
                "--constant",
                str(Path(OUTPUT_DIR) / f"{scale}_constant.json"),
                "--variable",
                str(Path(OUTPUT_DIR) / f"{scale}_variable.json"),
                "--out",
                str(comparison_path),
            ]
        )
        runs_volume.commit()
    return {
        "scale": scale,
        "model_kind": model_kind,
        "comparison": str(comparison_path) if comparison_path else None,
        "reports": {
            kind: {
                "path": str(Path(OUTPUT_DIR) / f"{scale}_{kind}.json"),
                "params": report["model"]["params"],
                "smoothed_final_loss": report["metrics"]["smoothed_final_loss"],
                "tokens_per_sec": report["metrics"]["tokens_per_sec"],
                "tokens_seen": report["data"]["tokens_seen"],
            }
            for kind, report in reports.items()
        },
    }


def run(cmd: list[str], commit_on_checkpoint: bool = False) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_DIR}/src"
    process = subprocess.Popen(
        cmd,
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if commit_on_checkpoint and line.startswith("checkpoint:"):
            runs_volume.commit()
            print("committed checkpoint volume", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def latest_checkpoint(checkpoint_dir: Path, scale: str, kind: str) -> Optional[Path]:
    prefix = f"{scale}_{kind}_step"
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{prefix}*.pt"):
        stem = path.stem
        try:
            step = int(stem.removeprefix(prefix))
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return max(checkpoints)[1]


@app.local_entrypoint()
def main(
    scale: str = "dense_200m",
    model_kind: str = "both",
    train_bin: str = f"{DATA_DIR}/dclm_cl100k_uint32.bin",
    checkpoint_interval: int = 1000,
    checkpoint_at_end: bool = True,
    keep_checkpoints: int = 2,
    resume: bool = True,
) -> None:
    result = run_paper_scale.remote(
        scale=scale,
        model_kind=model_kind,
        train_bin=train_bin,
        checkpoint_interval=checkpoint_interval,
        checkpoint_at_end=checkpoint_at_end,
        keep_checkpoints=keep_checkpoints,
        resume=resume,
    )
    print(json.dumps(result, indent=2))
