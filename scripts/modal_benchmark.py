#!/usr/bin/env python3
"""Run the benchmark on Modal GPU infrastructure.

Usage:
    modal run scripts/modal_benchmark.py --mode replicated
    modal run scripts/modal_benchmark.py --mode shape-sweep
    modal run scripts/modal_benchmark.py --mode white-nights-replicated
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


APP_DIR = "/root/vwt-bench"
OUTPUT_DIR = "/outputs"

app = modal.App("variable-width-transformer-bench")
volume = modal.Volume.from_name("vwt-bench-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy>=1.26")
    .add_local_dir("src", f"{APP_DIR}/src")
    .add_local_dir("data", f"{APP_DIR}/data")
    .add_local_dir("scripts", f"{APP_DIR}/scripts")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={OUTPUT_DIR: volume},
)
def run_remote(mode: str = "replicated") -> dict:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{mode.replace('-', '_')}.json"
    artifact_dir = output_dir / f"{mode.replace('-', '_')}_artifacts"

    benchmark_args = benchmark_command(mode, report_path)
    run(benchmark_args)
    run(
        [
            sys.executable,
            "scripts/build_artifacts.py",
            "--report",
            str(report_path),
            "--out-dir",
            str(artifact_dir),
            "--title",
            f"{mode.replace('-', ' ').title()} Variable-Width Benchmark",
        ]
    )
    volume.commit()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "report": str(report_path),
        "artifacts": str(artifact_dir),
        "summary": report.get("summary", []),
        "pairwise_vs_constant": report.get("pairwise_vs_constant", []),
    }


def benchmark_command(mode: str, report_path: Path) -> list[str]:
    if mode == "white-nights-replicated":
        return [
            sys.executable,
            "-m",
            "vwt_bench.benchmark",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--steps",
            "1000",
            "--eval-iters",
            "16",
            "--eval-interval",
            "200",
            "--history-interval",
            "10",
            "--log-interval",
            "50",
            "--seeds",
            "1337,2027,3141",
            "--layers",
            "16",
            "--width",
            "640",
            "--heads",
            "16",
            "--batch-size",
            "16",
            "--block-size",
            "512",
            "--generate-tokens",
            "400",
            "--prompt",
            "It was a wonderful night,",
            "--report-path",
            str(report_path),
        ]

    common = [
        sys.executable,
        "-m",
        "vwt_bench.benchmark",
        "--device",
        "cuda",
        "--steps",
        "500",
        "--eval-iters",
        "16",
        "--eval-interval",
        "100",
        "--history-interval",
        "5",
        "--layers",
        "6",
        "--width",
        "96",
        "--heads",
        "4",
        "--batch-size",
        "32",
        "--block-size",
        "96",
        "--generate-tokens",
        "240",
        "--report-path",
        str(report_path),
    ]
    if mode == "replicated":
        return common + ["--seeds", "1337,2027,3141"]
    if mode == "shape-sweep":
        return common + ["--variable-shapes", "x,diamond,increasing,decreasing"]
    raise ValueError("mode must be 'replicated', 'shape-sweep', or 'white-nights-replicated'")


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
def main(mode: str = "replicated") -> None:
    result = run_remote.remote(mode)
    print(json.dumps(result, indent=2))
