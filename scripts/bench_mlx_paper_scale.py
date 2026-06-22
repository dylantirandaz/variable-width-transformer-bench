#!/usr/bin/env python3
"""Bounded MLX training benchmark for the paper-scale VWT model.

This is a local Apple Silicon performance probe. It does not write the
pre-registered paper-scale comparison reports unless explicitly pointed at a
report path. By default it uses synthetic token batches so the model path can be
timed without downloading the 10B-token corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mx_utils
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when MLX is absent
    raise SystemExit("MLX is required. Install with: .venv/bin/python -m pip install mlx") from exc

from vwt_bench.paper_scale import PAPER_DENSE_SCALES, get_paper_scale
from vwt_bench.widths import geometric_widths, uniform_widths


DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}

TOKEN_DTYPES = {
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "int64": np.dtype("int64"),
}


def resize_residual(x: mx.array, out_dim: int, candidates: Iterable[mx.array]) -> mx.array:
    in_dim = x.shape[-1]
    if in_dim == out_dim:
        return x
    if in_dim > out_dim:
        return x[..., :out_dim]

    current = in_dim
    parts = []
    for candidate in candidates:
        candidate_dim = candidate.shape[-1]
        if candidate_dim <= current:
            continue
        end = min(candidate_dim, out_dim)
        parts.append(candidate[..., current:end])
        current = end
        if current >= out_dim:
            break

    needed = out_dim - in_dim
    if parts:
        expanded = mx.concatenate(parts, axis=-1)
        if expanded.shape[-1] < needed:
            pad = mx.zeros((*x.shape[:-1], needed - expanded.shape[-1]), dtype=x.dtype)
            expanded = mx.concatenate([expanded, pad], axis=-1)
    else:
        expanded = mx.zeros((*x.shape[:-1], needed), dtype=x.dtype)
    return mx.concatenate([x, expanded], axis=-1)


class MlxCausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, rope_base: float, attention_scale: str, dtype) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.rope_base = rope_base
        self.attention_scale = attention_scale
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.set_dtype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        batch, steps, width = x.shape
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (batch, steps, 3, self.heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        scale = 1.0 / self.head_dim if self.attention_scale == "mup" else 1.0 / (self.head_dim**0.5)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        y = mx.reshape(mx.transpose(y, (0, 2, 1, 3)), (batch, steps, width))
        return self.proj(y)


class MlxSwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int, dtype) -> None:
        super().__init__()
        inner = expansion * width
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)
        self.set_dtype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class MlxTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_expansion: int,
        rope_base: float,
        attention_scale: str,
        dtype,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(width)
        self.attn = MlxCausalSelfAttention(width, heads, rope_base, attention_scale, dtype)
        self.ln_2 = nn.RMSNorm(width)
        self.mlp = MlxSwiGLU(width, mlp_expansion, dtype)
        self.set_dtype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MlxTinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        base_width: int,
        widths: list[int],
        heads: int,
        mlp_expansion: int,
        rope_base: float,
        init_std: float,
        attention_scale: str,
        dtype,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.base_width = base_width
        self.widths = list(widths)
        self.dtype = dtype
        self.activation_checkpoint = activation_checkpoint
        self.token_embedding = nn.Embedding(vocab_size, base_width)
        self.blocks = [
            MlxTransformerBlock(
                width=width,
                heads=heads,
                mlp_expansion=mlp_expansion,
                rope_base=rope_base,
                attention_scale=attention_scale,
                dtype=dtype,
            )
            for width in widths
        ]
        self.ln_f = nn.RMSNorm(widths[-1])
        self.lm_head = nn.Linear(base_width, vocab_size, bias=False)
        self._init_weights(init_std)
        self.set_dtype(dtype)

    def _normal(self, shape, std: float) -> mx.array:
        return mx.random.normal(shape=shape, dtype=mx.float32) * std

    def _init_linear_or_embedding(self, module, std: float) -> None:
        module.weight = self._normal(module.weight.shape, std).astype(self.dtype)

    def _init_weights(self, init_std: float) -> None:
        self._init_linear_or_embedding(self.token_embedding, init_std)
        for width, block in zip(self.widths, self.blocks):
            std = init_std * ((self.base_width / width) ** 0.5)
            self._init_linear_or_embedding(block.attn.qkv, std)
            self._init_linear_or_embedding(block.attn.proj, std)
            self._init_linear_or_embedding(block.mlp.gate, std)
            self._init_linear_or_embedding(block.mlp.up, std)
            self._init_linear_or_embedding(block.mlp.down, std)
        self._init_linear_or_embedding(self.lm_head, init_std)

    def __call__(self, input_ids: mx.array, targets: mx.array, loss_chunk_size: int) -> mx.array:
        x = self.token_embedding(input_ids)
        histories = [x]
        for width, block in zip(self.widths, self.blocks):
            x = resize_residual(x, width, reversed(histories))
            if self.activation_checkpoint:
                x = mx.checkpoint(block)(x)
            else:
                x = block(x)
            histories.append(x)
        x = self.ln_f(x)
        x = resize_residual(x, self.base_width, reversed(histories))
        return chunked_cross_entropy(x, targets, self.lm_head.weight, loss_chunk_size)


def chunked_cross_entropy(hidden: mx.array, targets: mx.array, lm_weight: mx.array, chunk_size: int) -> mx.array:
    hidden = mx.reshape(hidden, (-1, hidden.shape[-1]))
    targets = mx.reshape(targets, (-1,))
    if chunk_size <= 0:
        logits = hidden @ mx.transpose(lm_weight)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    total = mx.array(0.0, dtype=mx.float32)
    count = hidden.shape[0]
    for start in range(0, count, chunk_size):
        h = hidden[start : start + chunk_size]
        y = targets[start : start + chunk_size]
        logits = h @ mx.transpose(lm_weight)
        total = total + nn.losses.cross_entropy(logits, y, reduction="sum")
    return total / count


class BatchSource:
    def __init__(
        self,
        *,
        train_bin: str | None,
        data_dtype: str,
        sequence_length: int,
        micro_batch_size: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        self.train_bin = train_bin
        self.sequence_length = sequence_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.tokens = None
        if train_bin:
            self.tokens = np.memmap(train_bin, mode="r", dtype=TOKEN_DTYPES[data_dtype])

    def batch(self) -> tuple[mx.array, mx.array]:
        if self.tokens is None:
            shape = (self.micro_batch_size, self.sequence_length)
            x_np = self.rng.integers(0, self.vocab_size, size=shape, dtype=np.int32)
            y_np = self.rng.integers(0, self.vocab_size, size=shape, dtype=np.int32)
        else:
            start = self.cursor
            token_count = self.micro_batch_size * self.sequence_length
            end = start + token_count + 1
            if end > len(self.tokens):
                self.cursor = 0
                start = 0
                end = token_count + 1
            window = np.asarray(self.tokens[start:end], dtype=np.int32)
            x_np = window[:-1].reshape(self.micro_batch_size, self.sequence_length)
            y_np = window[1:].reshape(self.micro_batch_size, self.sequence_length)
            self.cursor += token_count
        return mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32)


def count_params(params) -> int:
    total = 0
    for _, value in tree_items(params):
        total += int(value.size)
    return total


def tree_items(tree, prefix: str = ""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from tree_items(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from tree_items(value, f"{prefix}.{i}" if prefix else str(i))
    else:
        yield prefix, tree


def eval_tree(tree) -> None:
    arrays = [value for _, value in tree_items(tree)]
    if arrays:
        mx.eval(*arrays)


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    progress = min(max((step - args.warmup_steps) / max(args.decay_steps, 1), 0.0), 1.0)
    return args.lr * ((1.0 - progress) ** args.lr_decay_power)


def main() -> None:
    args = parse_args()
    if args.compile:
        raise SystemExit(
            "--compile is intentionally disabled for now: the mutating MLX "
            "optimizer step needs to be rewritten as a pure compiled function first."
        )
    mx.random.seed(args.seed)
    if args.memory_limit_gb > 0:
        mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    if args.cache_limit_gb >= 0:
        mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))

    scale = get_paper_scale(args.scale)
    if args.warmup_steps is None:
        args.warmup_steps = scale.warmup_steps
    if args.decay_steps is None:
        args.decay_steps = scale.decay_steps
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.paper_batch:
        tokens_per_micro = args.sequence_length * args.micro_batch_size
        if scale.tokens_per_step % tokens_per_micro != 0:
            raise ValueError(
                "--paper-batch requires scale.tokens_per_step to divide evenly by "
                "--sequence-length * --micro-batch-size"
            )
        args.gradient_accumulation_steps = scale.tokens_per_step // tokens_per_micro
    args.tokens_per_optimizer_step = (
        args.sequence_length * args.micro_batch_size * args.gradient_accumulation_steps
    )
    args.matches_paper_tokens_per_step = args.tokens_per_optimizer_step == scale.tokens_per_step

    dtype = DTYPES[args.dtype]
    schedule = (
        uniform_widths(scale.layers, scale.width)
        if args.model_kind == "constant"
        else geometric_widths(
            num_layers=scale.layers,
            base_width=scale.width,
            shape="x",
            bottleneck_layer_ratio=scale.bottleneck_layer_ratio,
            bottleneck_width_ratio=scale.bottleneck_width_ratio,
            quantize_to=scale.quantize_to,
            mlp_expansion=args.mlp_expansion,
        )
    )
    model = MlxTinyTransformerLM(
        vocab_size=scale.vocab_size,
        base_width=scale.width,
        widths=schedule.widths,
        heads=scale.heads,
        mlp_expansion=args.mlp_expansion,
        rope_base=args.rope_base,
        init_std=scale.init_std,
        attention_scale=scale.attention_scale,
        dtype=dtype,
        activation_checkpoint=args.activation_checkpoint,
    )
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    mx.eval(model.parameters(), optimizer.state)

    source = BatchSource(
        train_bin=args.train_bin,
        data_dtype=args.data_dtype,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        vocab_size=scale.vocab_size,
        seed=args.seed + 10_000,
    )

    def loss_fn(x, y):
        return model(x, y, args.loss_chunk_size)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_micro_step(x, y):
        loss, grads = loss_and_grad(x, y)
        mx.eval(loss)
        eval_tree(grads)
        return loss, grads

    def train_optimizer_step(step: int, started_at: float):
        optimizer.learning_rate = mx.array(learning_rate_for_step(args, step), dtype=mx.float32)
        loss_sum = mx.array(0.0, dtype=mx.float32)
        grads_sum = None
        micro_tokens = 0
        for _ in range(args.gradient_accumulation_steps):
            if args.max_seconds > 0 and time.perf_counter() - started_at >= args.max_seconds:
                return None, None, micro_tokens, False
            x, y = source.batch()
            loss, grads = train_micro_step(x, y)
            loss_sum = loss_sum + loss.astype(mx.float32)
            grads_sum = grads if grads_sum is None else mx_utils.tree_map(lambda a, b: a + b, grads_sum, grads)
            micro_tokens += args.micro_batch_size * args.sequence_length
            mx.eval(loss_sum)
            eval_tree(grads_sum)

        mean_loss = loss_sum / args.gradient_accumulation_steps
        mean_grads = mx_utils.tree_map(lambda g: g / args.gradient_accumulation_steps, grads_sum)
        grad_norm = mx.array(0.0, dtype=mx.float32)
        if args.grad_clip > 0:
            mean_grads, grad_norm = optim.clip_grad_norm(mean_grads, args.grad_clip)
        optimizer.update(model, mean_grads)
        mx.eval(mean_loss, grad_norm, model.parameters(), optimizer.state)
        return mean_loss, grad_norm, micro_tokens, True

    rows = []
    tokens_seen = 0
    started = time.perf_counter()
    last = started
    print_header(args, scale, schedule, model)
    for step in range(1, args.steps + 1):
        if args.max_seconds > 0 and time.perf_counter() - started >= args.max_seconds:
            print(f"stopping: max_seconds={args.max_seconds} reached", flush=True)
            break
        loss, grad_norm, micro_tokens, completed = train_optimizer_step(step, started)
        now = time.perf_counter()
        tokens_seen += micro_tokens
        if not completed:
            print("stopping: max_seconds reached before completing the next optimizer step", flush=True)
            break
        loss_value = float(loss.item())
        grad_norm_value = float(grad_norm.item()) if args.grad_clip > 0 else None
        row = {
            "step": step,
            "tokens": tokens_seen,
            "lr": float(optimizer.learning_rate.item()),
            "seconds": now - started,
            "step_seconds": now - last,
            "loss": loss_value,
            "grad_norm": grad_norm_value,
            "tokens_per_sec": tokens_seen / max(now - started, 1e-9),
            "active_memory": int(mx.get_active_memory()),
            "peak_memory": int(mx.get_peak_memory()),
            "cache_memory": int(mx.get_cache_memory()),
        }
        rows.append(row)
        last = now
        if step == 1 or step % args.log_interval == 0:
            print(
                f"step={step:05d} loss={row['loss']:.4f} "
                f"lr={row['lr']:.4g} "
                f"tok/s={row['tokens_per_sec']:.1f} "
                f"step_s={row['step_seconds']:.3f} "
                f"peak_gb={row['peak_memory'] / 1024**3:.2f}",
                flush=True,
            )
        if not math.isfinite(loss_value):
            print("stopping: non-finite loss", flush=True)
            break

    elapsed = time.perf_counter() - started
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "backend": "mlx",
        "scale": asdict(scale),
        "args": vars(args),
        "model": {
            "kind": args.model_kind,
            "params": count_params(model.parameters()),
            "widths": schedule.widths,
            "average_width": schedule.average_width,
        },
        "metrics": {
            "steps": len(rows),
            "tokens_seen": tokens_seen,
            "elapsed_seconds": elapsed,
            "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
            "peak_memory": int(mx.get_peak_memory()),
            "tokens_per_optimizer_step": args.tokens_per_optimizer_step,
            "matches_paper_tokens_per_step": args.matches_paper_tokens_per_step,
        },
        "history": rows,
    }
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report: {path}", flush=True)


def print_header(args, scale, schedule, model) -> None:
    print(f"backend=mlx scale={scale.name} model={args.model_kind} dtype={args.dtype}")
    print(f"seq={args.sequence_length} activation_checkpoint={args.activation_checkpoint} compile={args.compile}")
    print(
        f"micro_batch={args.micro_batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"tokens/update={args.tokens_per_optimizer_step:,} "
        f"paper_batch={args.matches_paper_tokens_per_step}"
    )
    print(f"loss_chunk_size={args.loss_chunk_size} train_bin={args.train_bin or 'synthetic'}")
    print(f"widths={schedule.widths}")
    print(f"params={count_params(model.parameters()):,} avg_width={schedule.average_width:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(PAPER_DENSE_SCALES), default="dense_200m")
    parser.add_argument("--model-kind", choices=["constant", "variable"], default="variable")
    parser.add_argument("--train-bin", default=None)
    parser.add_argument("--data-dtype", choices=sorted(TOKEN_DTYPES), default="uint32")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--paper-batch",
        action="store_true",
        help="Accumulate enough local microbatches to match the scale's paper tokens/update.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--report-path", default="runs/paper_scale/mlx_probe.json")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-chunk-size", type=int, default=1024)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--lr-decay-power", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
