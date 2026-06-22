#!/usr/bin/env python3
"""Train a paper-scale dense model from a pre-tokenized cl100k memmap.

This is the large-token path for this repo. It intentionally avoids the tiny
benchmark's random reusable batches: every optimizer step consumes the next
contiguous shard of the token stream.

Launch the 200M setting on an 8-GPU node with:

    torchrun --standalone --nproc_per_node=8 scripts/train_paper_scale.py \
      --scale dense_200m \
      --model-kind variable \
      --train-bin /data/dclm_cl100k_uint32.bin
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from vwt_bench.model import TinyTransformerLM, count_parameters
from vwt_bench.paper_scale import PAPER_DENSE_SCALES, get_paper_scale
from vwt_bench.token_data import MemmapTokenDataset
from vwt_bench.widths import geometric_widths, uniform_widths


def main() -> None:
    args = parse_args()
    dist_state = setup_distributed()
    device = select_device(args.device, dist_state["local_rank"])
    is_rank0 = dist_state["rank"] == 0

    scale = get_paper_scale(args.scale)
    args = apply_scale_defaults(args, scale)
    validate_runtime(args, dist_state)

    if args.seed is not None:
        torch.manual_seed(args.seed + dist_state["rank"])

    dataset = MemmapTokenDataset(args.train_bin, dtype=args.data_dtype)
    dataset.require_tokens(args.steps * args.tokens_per_step)

    schedule = (
        uniform_widths(args.layers, args.width)
        if args.model_kind == "constant"
        else geometric_widths(
            num_layers=args.layers,
            base_width=args.width,
            shape="x",
            bottleneck_layer_ratio=args.bottleneck_layer_ratio,
            bottleneck_width_ratio=args.bottleneck_width_ratio,
            quantize_to=args.quantize_to,
            mlp_expansion=args.mlp_expansion,
            endpoint_correction=not args.disable_endpoint_correction,
        )
    )

    model = TinyTransformerLM(
        vocab_size=args.vocab_size,
        block_size=args.sequence_length,
        base_width=args.width,
        widths=schedule.widths,
        heads=args.heads,
        dropout=args.dropout,
        mlp_expansion=args.mlp_expansion,
        position_encoding="rope",
        rope_base=args.rope_base,
        init_std=args.init_std,
        width_aware_init=not args.disable_width_aware_init,
        norm=args.norm,
        attention_scale=args.attention_scale,
    ).to(device)

    if dist_state["distributed"]:
        model_for_train = DistributedDataParallel(
            model,
            device_ids=[device.index],
            gradient_as_bucket_view=True,
        )
    else:
        model_for_train = model

    optimizer = torch.optim.AdamW(
        model_for_train.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_step = 1
    if args.resume:
        checkpoint_step = load_checkpoint(args.resume, model, optimizer, device)
        if checkpoint_step >= args.steps:
            raise ValueError(
                f"checkpoint step {checkpoint_step} is already at or beyond requested --steps {args.steps}"
            )
        start_step = checkpoint_step + 1
        if is_rank0:
            print(f"resumed checkpoint={args.resume} next_step={start_step}", flush=True)

    if is_rank0:
        print_run_header(args, scale, schedule, model, dist_state, dataset)

    history: list[dict[str, float]] = []
    smoothing_losses: list[float] = []
    tokens_seen = (start_step - 1) * args.tokens_per_step
    started = time.perf_counter()
    model_for_train.train()

    for step in range(start_step, args.steps + 1):
        lr = learning_rate_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        for accum_idx in range(args.gradient_accumulation_steps):
            start_sequence = (
                (step - 1) * args.sequences_per_step
                + accum_idx * dist_state["world_size"] * args.micro_batch_size
                + dist_state["rank"] * args.micro_batch_size
            )
            x, y = dataset.sequential_batch(
                start_sequence=start_sequence,
                batch_size=args.micro_batch_size,
                sequence_length=args.sequence_length,
                device=device,
            )
            should_sync = accum_idx == args.gradient_accumulation_steps - 1
            sync_context = (
                nullcontext()
                if should_sync or not dist_state["distributed"]
                else model_for_train.no_sync()
            )
            with sync_context:
                with precision_context(args, device):
                    _, loss = model_for_train(x, y, loss_chunk_size=args.loss_chunk_size)
                raw_loss = float(loss.item())
                (loss / args.gradient_accumulation_steps).backward()
            step_loss_sum += raw_loss

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), args.grad_clip)
        optimizer.step()

        step_loss = distributed_average(
            step_loss_sum / args.gradient_accumulation_steps,
            device,
            dist_state["distributed"],
        )
        tokens_seen = step * args.tokens_per_step
        elapsed = elapsed_seconds(started, device)
        row = {
            "step": float(step),
            "tokens": float(tokens_seen),
            "loss": step_loss,
            "lr": lr,
            "seconds": elapsed,
            "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
        }

        in_smoothing_window = step > args.steps - args.loss_window_steps
        if in_smoothing_window and step % args.loss_sample_interval == 0:
            smoothing_losses.append(step_loss)

        should_log = step == start_step or step == args.steps or (
            args.log_interval > 0 and step % args.log_interval == 0
        )
        if is_rank0 and should_log:
            history.append(row)
            print(
                f"step={step:05d}/{args.steps} "
                f"loss={step_loss:.4f} lr={lr:.4g} "
                f"tokens={tokens_seen:,} tok/s={row['tokens_per_sec']:.0f}",
                flush=True,
            )

        if is_rank0 and args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save_checkpoint(args, model, optimizer, step)

    final_loss = smoothing_losses[-1] if smoothing_losses else history[-1]["loss"]
    smoothed_final_loss = sum(smoothing_losses) / len(smoothing_losses) if smoothing_losses else final_loss
    if is_rank0:
        report = build_report(
            args=args,
            scale=scale,
            schedule=schedule,
            model=model,
            dist_state=dist_state,
            history=history,
            final_loss=final_loss,
            smoothed_final_loss=smoothed_final_loss,
            tokens_seen=tokens_seen,
            elapsed=elapsed_seconds(started, device),
        )
        write_report(args.report_path, report)
        if args.checkpoint_at_end:
            save_checkpoint(args, model, optimizer, args.steps)

    cleanup_distributed(dist_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(PAPER_DENSE_SCALES), default="dense_200m")
    parser.add_argument("--model-kind", choices=["constant", "variable"], default="variable")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--data-dtype", default="uint32", choices=["uint16", "uint32", "int64"])
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--checkpoint-dir", default="runs/paper_scale/checkpoints")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save a rank-0 checkpoint every N steps; 0 disables interval checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a final rank-0 checkpoint after writing the report.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Keep this many newest checkpoints for this scale/model; 0 keeps all.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume model and optimizer state from a checkpoint path.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tokens-per-step", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--allow-batch-rescale", action="store_true")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--bottleneck-layer-ratio", type=float, default=None)
    parser.add_argument("--bottleneck-width-ratio", type=float, default=None)
    parser.add_argument("--quantize-to", type=int, default=None)
    parser.add_argument("--disable-endpoint-correction", action="store_true")
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--norm", choices=["layernorm", "rmsnorm"], default=None)
    parser.add_argument("--attention-scale", choices=["sqrt", "mup"], default=None)
    parser.add_argument("--init-std", type=float, default=None)
    parser.add_argument("--disable-width-aware-init", action="store_true")
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--lr-decay-power", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--loss-window-steps", type=int, default=1000)
    parser.add_argument("--loss-sample-interval", type=int, default=10)
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=1024,
        help="Token chunk size for checkpointed vocab loss; 0 materializes full logits.",
    )
    return parser.parse_args()


def apply_scale_defaults(args: argparse.Namespace, scale) -> argparse.Namespace:
    for attr, value in {
        "steps": scale.training_steps,
        "tokens_per_step": scale.tokens_per_step,
        "sequence_length": scale.sequence_length,
        "micro_batch_size": scale.micro_batch_size,
        "gradient_accumulation_steps": scale.gradient_accumulation_steps,
        "layers": scale.layers,
        "width": scale.width,
        "heads": scale.heads,
        "vocab_size": scale.vocab_size,
        "bottleneck_layer_ratio": scale.bottleneck_layer_ratio,
        "bottleneck_width_ratio": scale.bottleneck_width_ratio,
        "quantize_to": scale.quantize_to,
        "norm": scale.norm,
        "attention_scale": scale.attention_scale,
        "init_std": scale.init_std,
        "warmup_steps": scale.warmup_steps,
        "decay_steps": scale.decay_steps,
    }.items():
        if getattr(args, attr) is None:
            setattr(args, attr, value)
    args.sequences_per_step = args.tokens_per_step // args.sequence_length
    if args.report_path is None:
        args.report_path = f"runs/paper_scale/{args.scale}_{args.model_kind}.json"
    return args


def setup_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def cleanup_distributed(dist_state: dict[str, Any]) -> None:
    if dist_state["distributed"]:
        torch.distributed.destroy_process_group()


def select_device(choice: str, local_rank: int) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def validate_runtime(args: argparse.Namespace, dist_state: dict[str, Any]) -> None:
    if args.precision == "bf16" and args.device != "cuda":
        raise RuntimeError("bf16 paper-scale training requires CUDA")
    if args.tokens_per_step % args.sequence_length != 0:
        raise ValueError("--tokens-per-step must be divisible by --sequence-length")
    expected_sequences = args.micro_batch_size * args.gradient_accumulation_steps * dist_state["world_size"]
    if expected_sequences != args.sequences_per_step:
        if not args.allow_batch_rescale:
            raise ValueError(
                "effective batch does not match the paper-scale token batch: "
                f"micro_batch_size({args.micro_batch_size}) * "
                f"gradient_accumulation_steps({args.gradient_accumulation_steps}) * "
                f"world_size({dist_state['world_size']}) = {expected_sequences} sequences, "
                f"but tokens_per_step/sequence_length = {args.sequences_per_step}. "
                "Use the official 8-GPU launch or pass --allow-batch-rescale deliberately."
            )
        args.sequences_per_step = expected_sequences
        args.tokens_per_step = expected_sequences * args.sequence_length
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    if args.loss_sample_interval <= 0:
        raise ValueError("--loss-sample-interval must be positive")


def precision_context(args: argparse.Namespace, device: torch.device):
    if args.precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if step <= args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    progress = min(max((step - args.warmup_steps) / max(args.decay_steps, 1), 0.0), 1.0)
    return args.lr * ((1.0 - progress) ** args.lr_decay_power)


def distributed_average(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return float(tensor.item())


def elapsed_seconds(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return max(time.perf_counter() - started, 1e-9)


def load_checkpoint(path: str, model: TinyTransformerLM, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def save_checkpoint(args: argparse.Namespace, model: TinyTransformerLM, optimizer: torch.optim.Optimizer, step: int) -> None:
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.scale}_{args.model_kind}_step{step:06d}.pt"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp_path,
    )
    tmp_path.replace(path)
    prune_checkpoints(args, out_dir)
    print(f"checkpoint: {path}", flush=True)


def prune_checkpoints(args: argparse.Namespace, out_dir: Path) -> None:
    if args.keep_checkpoints <= 0:
        return
    prefix = f"{args.scale}_{args.model_kind}_step"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.pt$")
    checkpoints: list[tuple[int, Path]] = []
    for path in out_dir.glob(f"{prefix}*.pt"):
        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(reverse=True)
    for _, path in checkpoints[args.keep_checkpoints :]:
        path.unlink(missing_ok=True)


def build_report(
    args: argparse.Namespace,
    scale,
    schedule,
    model: TinyTransformerLM,
    dist_state: dict[str, Any],
    history: list[dict[str, float]],
    final_loss: float,
    smoothed_final_loss: float,
    tokens_seen: int,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scale": asdict(scale),
        "args": vars(args),
        "distributed": dist_state,
        "model": {
            "kind": args.model_kind,
            "params": count_parameters(model),
            "widths": schedule.widths,
            "average_width": schedule.average_width,
            "square_sum": schedule.square_sum,
            "target_square_sum": schedule.target_square_sum,
        },
        "data": {
            "train_bin": args.train_bin,
            "tokens_seen": tokens_seen,
            "tokens_per_step": args.tokens_per_step,
            "sequence_length": args.sequence_length,
            "sequences_per_step": args.sequences_per_step,
        },
        "metrics": {
            "final_loss": final_loss,
            "smoothed_final_loss": smoothed_final_loss,
            "elapsed_seconds": elapsed,
            "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
        },
        "history": history,
    }


def write_report(path: str, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}", flush=True)


def print_run_header(args, scale, schedule, model, dist_state, dataset) -> None:
    print(f"scale={scale.name} nominal={scale.nominal_params} model={args.model_kind}")
    print(f"distributed world_size={dist_state['world_size']} micro_batch={args.micro_batch_size} accum={args.gradient_accumulation_steps}")
    print(f"tokens: dataset={len(dataset):,} step={args.tokens_per_step:,} total={args.steps * args.tokens_per_step:,}")
    print(f"shape: layers={args.layers} width={args.width} heads={args.heads} seq={args.sequence_length}")
    print(f"widths={schedule.widths}")
    print(f"params={count_parameters(model):,} avg_width={schedule.average_width:.1f}")


if __name__ == "__main__":
    main()
