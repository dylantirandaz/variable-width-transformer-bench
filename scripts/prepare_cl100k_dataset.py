#!/usr/bin/env python3
"""Encode text corpora into a flat cl100k token memmap.

``--max-tokens`` counts stored memmap tokens. For next-token training, prepare
one more token than the number of input tokens the schedule will consume.

Examples:
    python scripts/prepare_cl100k_dataset.py \
      --input data/*.jsonl \
      --text-key text \
      --output data/dclm_cl100k_uint32.bin \
      --max-tokens 10003415041

    python scripts/prepare_cl100k_dataset.py \
      --hf-dataset mlfoundations/dclm-baseline-1.0 \
      --hf-split train \
      --output data/dclm_cl100k_uint32.bin \
      --max-tokens 10003415041
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import time
from typing import Iterable, Iterator

import numpy as np

from vwt_bench.paper_scale import CL100K_EOS_TOKEN, CL100K_PAPER_VOCAB_SIZE


def main() -> None:
    args = parse_args()
    encoder = load_encoder()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    docs = 0
    tokens = 0
    with output.open("wb") as writer:
        for text in iter_texts(args):
            ids = encoder.encode(text, disallowed_special=())
            if args.append_eos:
                ids.append(args.eos_token)
            if not ids:
                continue
            remaining = args.max_tokens - tokens if args.max_tokens else None
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and len(ids) > remaining:
                ids = ids[:remaining]
            np.asarray(ids, dtype=np.uint32).tofile(writer)
            docs += 1
            tokens += len(ids)
            if args.log_interval > 0 and docs % args.log_interval == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"docs={docs:,} tokens={tokens:,} tok/s={tokens / elapsed:,.0f}", flush=True)
            if args.max_tokens and tokens >= args.max_tokens:
                break

    manifest = {
        "output": str(output),
        "tokenizer": "cl100k_base",
        "vocab_size": CL100K_PAPER_VOCAB_SIZE,
        "dtype": "uint32",
        "tokens": tokens,
        "documents": docs,
        "append_eos": args.append_eos,
        "eos_token": args.eos_token,
        "max_tokens": args.max_tokens,
        "inputs": args.input,
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"wrote {tokens:,} tokens from {docs:,} docs to {output} in {elapsed:.1f}s")
    print(f"manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", default=[], help="Text or JSONL files/globs.")
    parser.add_argument("--text-key", default="text", help="JSON/JSONL field containing document text.")
    parser.add_argument("--hf-dataset", default=None, help="Optional Hugging Face dataset name.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-tokens", type=int, default=0, help="Stop after this many tokens; 0 means no cap.")
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eos-token", type=int, default=CL100K_EOS_TOKEN)
    parser.add_argument("--log-interval", type=int, default=10_000)
    args = parser.parse_args()
    if not args.input and not args.hf_dataset:
        raise ValueError("provide --input files/globs or --hf-dataset")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens must be non-negative")
    return args


def load_encoder():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "prepare_cl100k_dataset.py requires tiktoken; install requirements-scale.txt"
        ) from exc
    return tiktoken.get_encoding("cl100k_base")


def iter_texts(args: argparse.Namespace) -> Iterator[str]:
    yield from iter_local_texts(args.input, args.text_key)
    if args.hf_dataset:
        yield from iter_hf_texts(args)


def iter_local_texts(patterns: Iterable[str], text_key: str) -> Iterator[str]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(text_key)
                    if isinstance(text, str):
                        yield text
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get(text_key), str):
                    yield row[text_key]
        else:
            yield path.read_text(encoding="utf-8")


def iter_hf_texts(args: argparse.Namespace) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face dataset streaming requires datasets; install requirements-scale.txt"
        ) from exc

    dataset_kwargs = {
        "path": args.hf_dataset,
        "name": args.hf_config,
        "split": args.hf_split,
        "streaming": args.hf_streaming,
    }
    if args.hf_config is None:
        dataset_kwargs.pop("name")
    dataset = load_dataset(**dataset_kwargs)
    for row in dataset:
        text = row.get(args.text_key)
        if isinstance(text, str):
            yield text


if __name__ == "__main__":
    main()
