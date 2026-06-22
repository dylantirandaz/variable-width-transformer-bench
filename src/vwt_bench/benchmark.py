"""CLI benchmark for constant-width vs variable-width Transformers."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import textwrap
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from vwt_bench.data import VOCAB_SIZE, decode, encode, get_batch, load_bytes, train_val_split
from vwt_bench.model import TinyTransformerLM, count_parameters, resize_residual
from vwt_bench.widths import WidthSchedule, geometric_widths, uniform_widths


@dataclass
class Result:
    name: str
    shape: str
    seed: int
    widths: list[int]
    params: int
    average_width: float
    square_sum: int
    target_square_sum: int
    final_train_loss: float
    val_loss: float
    val_ppl: float
    best_val_loss: float
    train_seconds: float
    tokens_per_sec: float
    efficiency: dict[str, float]
    diagnostics: dict[str, Any]
    history: list[dict[str, float]]
    generation: str


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    validate_precision(args, device)
    torch.manual_seed(args.seed)

    data = load_bytes(args.data_path)
    train_data, val_data = train_val_split(data, args.val_fraction)

    schedules = build_schedules(args)
    seeds = parse_seed_list(args.seeds) if args.seeds else [args.seed]

    print(f"device: {device}")
    print(f"tokens: train={len(train_data):,}, val={len(val_data):,}")
    for name, _, schedule in schedules:
        print_schedule(name, schedule)
    print()

    results = []
    for seed in seeds:
        run_args = copy.copy(args)
        run_args.seed = seed
        print_seed_protocol(seed)
        for name, shape, schedule in schedules:
            results.append(run_one(name, shape, schedule, run_args, train_data, val_data, device))

    print_summary(results)
    if len(results) > len(schedules):
        print_group_summary(summarize_results(results))
    print_pairwise_summary(pairwise_comparisons(results))
    first_by_name = {result.name: result for result in results}
    if "constant" in first_by_name and "variable" in first_by_name:
        print_generations(first_by_name["constant"].generation, first_by_name["variable"].generation)
    write_report(args.report_path, args, train_data, val_data, device, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None, help="UTF-8 text file. Defaults to data/tiny_corpus.txt.")
    parser.add_argument("--report-path", default="runs/last_run.json", help="Write a JSON report here.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for replicated benchmark runs.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=0, help="Also estimate validation loss every N train steps.")
    parser.add_argument("--history-interval", type=int, default=1, help="Record train-curve history every N steps.")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--lr-decay-power", type=float, default=1.0)
    parser.add_argument("--disable-lr-schedule", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--position-encoding", default="rope", choices=["rope", "learned"])
    parser.add_argument("--norm", default="layernorm", choices=["layernorm", "rmsnorm"])
    parser.add_argument("--attention-scale", default="sqrt", choices=["sqrt", "mup"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--disable-width-aware-init", action="store_true")
    parser.add_argument("--analysis-iters", type=int, default=1, help="Validation batches for representation diagnostics.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--bottleneck-layer-ratio", type=float, default=0.75)
    parser.add_argument("--bottleneck-width-ratio", type=float, default=0.30)
    parser.add_argument(
        "--variable-shapes",
        default="x",
        help="Comma-separated variable schedules: x, diamond, increasing, decreasing.",
    )
    parser.add_argument("--disable-endpoint-correction", action="store_true")
    parser.add_argument("--prompt", default="Tell me, ")
    parser.add_argument("--generate-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if choice == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(choice)


def validate_precision(args: argparse.Namespace, device: torch.device) -> None:
    if args.precision == "bf16" and device.type != "cuda":
        raise RuntimeError("--precision bf16 is currently only enabled for CUDA runs")


def precision_context(args: argparse.Namespace, device: torch.device):
    if args.precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def print_schedule(name: str, schedule: WidthSchedule) -> None:
    print(
        f"{name:>8} widths={schedule.widths} "
        f"avg={schedule.average_width:.1f} "
        f"sum_w2={schedule.square_sum:,}/{schedule.target_square_sum:,}"
    )
    if name != "constant":
        print(
            f"{'':>8} l*={schedule.bottleneck_layer} "
            f"target_bottleneck={schedule.target_bottleneck:.1f} "
            f"alpha_down={schedule.alpha_down:.4f} alpha_up={schedule.alpha_up:.4f}"
        )


def build_schedules(args: argparse.Namespace) -> list[tuple[str, str, WidthSchedule]]:
    schedules = [("constant", "constant", uniform_widths(args.layers, args.width))]
    shapes = parse_shape_list(args.variable_shapes)
    for shape in shapes:
        name = "variable" if shapes == ["x"] else f"variable_{shape}"
        schedules.append(
            (
                name,
                shape,
                geometric_widths(
                    num_layers=args.layers,
                    base_width=args.width,
                    shape=shape,
                    bottleneck_layer_ratio=args.bottleneck_layer_ratio,
                    bottleneck_width_ratio=args.bottleneck_width_ratio,
                    quantize_to=width_quantum(args.heads, args.position_encoding),
                    mlp_expansion=args.mlp_expansion,
                    endpoint_correction=not args.disable_endpoint_correction,
                ),
            )
        )
    return schedules


def parse_shape_list(raw: str) -> list[str]:
    shapes = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"x", "diamond", "increasing", "decreasing"}
    unknown = sorted(set(shapes) - allowed)
    if unknown:
        raise ValueError(f"unknown variable shape(s): {', '.join(unknown)}")
    return shapes or ["x"]


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def print_seed_protocol(seed: int) -> None:
    print(
        "seeds: "
        f"model={seed} "
        f"train_batches={seed + 10_000} "
        f"eval_batches={seed + 20_000} "
        f"sampling={seed + 30_000}"
    )


def width_quantum(heads: int, position_encoding: str) -> int:
    if position_encoding == "rope":
        return heads * 2
    return heads


def run_one(
    name: str,
    shape: str,
    schedule: WidthSchedule,
    args: argparse.Namespace,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
) -> Result:
    torch.manual_seed(args.seed)
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        block_size=args.block_size,
        base_width=args.width,
        widths=schedule.widths,
        heads=args.heads,
        dropout=args.dropout,
        mlp_expansion=args.mlp_expansion,
        position_encoding=args.position_encoding,
        rope_base=args.rope_base,
        init_std=args.init_std,
        width_aware_init=not args.disable_width_aware_init,
        norm=args.norm,
        attention_scale=args.attention_scale,
    ).to(device)
    params = count_parameters(model)
    efficiency = estimate_efficiency(schedule, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    batch_generator = torch.Generator().manual_seed(args.seed + 10_000)

    model.train()
    last_loss = float("nan")
    tokens_seen = 0
    eval_seconds = 0.0
    history: list[dict[str, float]] = []
    final_val_loss: Optional[float] = None
    synchronize_device(device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = learning_rate_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.block_size, device, batch_generator)
        with precision_context(args, device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
        tokens_seen += args.batch_size * args.block_size

        should_log = args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == args.steps)
        should_eval = args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.steps)
        should_record = step == 1 or step == args.steps or (
            args.history_interval > 0 and step % args.history_interval == 0
        )
        entry: Optional[dict[str, float]] = None

        if should_log or should_record or should_eval:
            elapsed = elapsed_train_seconds(start, eval_seconds, device)
            entry = {
                "step": float(step),
                "tokens": float(tokens_seen),
                "seconds": elapsed,
                "train_loss": last_loss,
                "tokens_per_sec": tokens_seen / elapsed,
                "lr": lr,
            }

        if should_eval:
            eval_start = time.perf_counter()
            val_loss = estimate_loss(model, val_data, args, device, seed=args.seed + 20_000)
            synchronize_device(device)
            eval_seconds += time.perf_counter() - eval_start
            final_val_loss = val_loss
            if entry is None:
                elapsed = elapsed_train_seconds(start, eval_seconds, device)
                entry = {
                    "step": float(step),
                    "tokens": float(tokens_seen),
                    "seconds": elapsed,
                    "train_loss": last_loss,
                    "tokens_per_sec": tokens_seen / elapsed,
                    "lr": lr,
                }
            entry["val_loss"] = val_loss
            entry["val_ppl"] = math.exp(min(val_loss, 20.0))

        if entry is not None and (should_record or should_eval):
            history.append(entry)

        if should_log:
            val_part = f" val={entry['val_loss']:.4f}" if entry is not None and "val_loss" in entry else ""
            print(
                f"{name:>8} step={step:04d}/{args.steps} "
                f"loss={last_loss:.4f}{val_part} tok/s={entry['tokens_per_sec']:.0f}"
            )

    train_seconds = elapsed_train_seconds(start, eval_seconds, device)
    if final_val_loss is None:
        final_val_loss = estimate_loss(model, val_data, args, device, seed=args.seed + 20_000)
    val_loss = final_val_loss
    if history:
        if int(history[-1]["step"]) == args.steps:
            history[-1]["val_loss"] = val_loss
            history[-1]["val_ppl"] = math.exp(min(val_loss, 20.0))
        else:
            history.append(
                {
                    "step": float(args.steps),
                    "tokens": float(tokens_seen),
                    "seconds": train_seconds,
                    "train_loss": last_loss,
                    "tokens_per_sec": tokens_seen / train_seconds,
                    "lr": learning_rate_for_step(args, args.steps),
                    "val_loss": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20.0)),
                }
            )
    else:
        history.append(
            {
                "step": float(args.steps),
                "tokens": float(tokens_seen),
                "seconds": train_seconds,
                "train_loss": last_loss,
                "tokens_per_sec": tokens_seen / train_seconds,
                "lr": learning_rate_for_step(args, args.steps),
                "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20.0)),
            }
        )
    best_val_loss = min(entry["val_loss"] for entry in history if "val_loss" in entry)
    diagnostics = analyze_representations(model, val_data, args, device, seed=args.seed + 40_000)
    generation = generate_text(model, args, device, seed=args.seed + 30_000)
    return Result(
        name=name,
        shape=shape,
        seed=args.seed,
        widths=schedule.widths,
        params=params,
        average_width=schedule.average_width,
        square_sum=schedule.square_sum,
        target_square_sum=schedule.target_square_sum,
        final_train_loss=last_loss,
        val_loss=val_loss,
        val_ppl=math.exp(min(val_loss, 20.0)),
        best_val_loss=best_val_loss,
        train_seconds=train_seconds,
        tokens_per_sec=tokens_seen / train_seconds,
        efficiency=efficiency,
        diagnostics=diagnostics,
        history=history,
        generation=generation,
    )


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLM,
    data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    losses = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, device, generator)
        with precision_context(args, device):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if args.disable_lr_schedule:
        return args.lr
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if args.lr_decay_power <= 0:
        raise ValueError("lr_decay_power must be positive")

    warmup_steps = int(args.steps * args.warmup_fraction)
    if warmup_steps > 0 and step <= warmup_steps:
        return args.lr * step / warmup_steps

    decay_steps = max(args.steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return args.lr * ((1.0 - progress) ** args.lr_decay_power)


def estimate_efficiency(schedule: WidthSchedule, args: argparse.Namespace) -> dict[str, float]:
    sum_width = float(sum(schedule.widths))
    baseline_sum_width = float(args.layers * args.width)
    linear_projection_proxy = float((4 + 3 * args.mlp_expansion) * schedule.square_sum)
    baseline_linear_projection_proxy = float((4 + 3 * args.mlp_expansion) * schedule.target_square_sum)
    attention_proxy = float(args.block_size * args.block_size * sum_width)
    baseline_attention_proxy = float(args.block_size * args.block_size * baseline_sum_width)
    kv_cache_elements_per_token = float(2 * sum_width)
    baseline_kv_cache_elements_per_token = float(2 * baseline_sum_width)
    return {
        "sum_width": sum_width,
        "average_width_ratio": schedule.average_width / args.width,
        "linear_projection_proxy": linear_projection_proxy,
        "linear_projection_proxy_ratio": linear_projection_proxy / baseline_linear_projection_proxy,
        "attention_proxy": attention_proxy,
        "attention_proxy_ratio": attention_proxy / baseline_attention_proxy,
        "kv_cache_elements_per_token": kv_cache_elements_per_token,
        "kv_cache_ratio": kv_cache_elements_per_token / baseline_kv_cache_elements_per_token,
    }


@torch.no_grad()
def analyze_representations(
    model: TinyTransformerLM,
    data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if args.analysis_iters <= 0:
        return {}
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    accum: list[dict[str, float]] = []
    for _ in range(args.analysis_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, device, generator)
        with precision_context(args, device):
            _, _, diagnostics = model(x, y, return_diagnostics=True)
        hidden_states = diagnostics["hidden_states"]
        for layer_idx, hidden in enumerate(hidden_states):
            metrics = representation_metrics(hidden)
            resized = resize_residual(hidden, model.base_width, reversed(hidden_states[:layer_idx]))
            with precision_context(args, device):
                logits = model.lm_head(resized)
                lens_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            accum.append(
                {
                    "layer": float(layer_idx),
                    "width": float(hidden.shape[-1]),
                    "participation_ratio": metrics["participation_ratio"],
                    "participation_ratio_fraction": metrics["participation_ratio_fraction"],
                    "logit_lens_loss": float(lens_loss.item()),
                }
            )
    model.train()
    return {"layers": average_layer_metrics(accum)}


def representation_metrics(hidden: torch.Tensor) -> dict[str, float]:
    flat = hidden.detach().float().reshape(-1, hidden.shape[-1])
    energy = flat.square().sum(dim=0)
    denom = energy.square().sum().clamp_min(1e-12)
    pr = float((energy.sum().square() / denom).item())
    return {
        "participation_ratio": pr,
        "participation_ratio_fraction": pr / hidden.shape[-1],
    }


def average_layer_metrics(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_layer: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    out = []
    for layer, layer_rows in sorted(by_layer.items()):
        out.append(
            {
                "layer": layer,
                "width": int(layer_rows[0]["width"]),
                "participation_ratio": mean(row["participation_ratio"] for row in layer_rows),
                "participation_ratio_fraction": mean(row["participation_ratio_fraction"] for row in layer_rows),
                "logit_lens_loss": mean(row["logit_lens_loss"] for row in layer_rows),
            }
        )
    return out


@torch.no_grad()
def generate_text(
    model: TinyTransformerLM,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> str:
    model.eval()
    torch.manual_seed(seed)
    prompt = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    generator: Optional[torch.Generator]
    if device.type == "cpu":
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    with precision_context(args, device):
        out = model.generate(
            prompt,
            max_new_tokens=args.generate_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            generator=generator,
        )
    return decode(out[0].detach().cpu().tolist())


def print_summary(results: list[Result]) -> None:
    print("\nsummary")
    header = (
        f"{'model':<18} {'seed':>6} {'params':>10} {'avg_w':>8} {'kv':>6} "
        f"{'tok/s':>9} {'train':>8} {'val':>8} {'best':>8} {'ppl':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<18} "
            f"{result.seed:>6} "
            f"{result.params:>10,} "
            f"{result.average_width:>8.1f} "
            f"{result.efficiency['kv_cache_ratio']:>6.2f} "
            f"{result.tokens_per_sec:>9.0f} "
            f"{result.final_train_loss:>8.4f} "
            f"{result.val_loss:>8.4f} "
            f"{result.best_val_loss:>8.4f} "
            f"{result.val_ppl:>8.2f}"
        )


def print_group_summary(rows: list[dict[str, Any]]) -> None:
    print("\naggregate")
    header = f"{'model':<18} {'runs':>4} {'val_mean':>9} {'val_sd':>8} {'best_mean':>9} {'tok/s':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<18} {row['runs']:>4} "
            f"{row['val_loss_mean']:>9.4f} {row['val_loss_stdev']:>8.4f} "
            f"{row['best_val_loss_mean']:>9.4f} {row['tokens_per_sec_mean']:>9.0f}"
        )


def print_pairwise_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("\npairwise vs constant")
    header = f"{'model':<18} {'runs':>4} {'val_delta':>10} {'best_delta':>11} {'tok/s_delta':>12} {'wins':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<18} {row['runs']:>4} "
            f"{row['val_loss_delta_mean']:>10.4f} {row['best_val_loss_delta_mean']:>11.4f} "
            f"{row['tokens_per_sec_delta_pct_mean']:>11.1f}% {row['val_loss_wins']:>6}"
        )


def print_generations(constant: str, variable: str) -> None:
    width = 58
    left_lines = textwrap.wrap(constant, width=width, replace_whitespace=False) or [""]
    right_lines = textwrap.wrap(variable, width=width, replace_whitespace=False) or [""]
    rows = max(len(left_lines), len(right_lines))
    print("\ngenerations")
    print(f"{'constant':<{width}} | variable")
    print(f"{'-' * width}-+-{'-' * width}")
    for i in range(rows):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{left:<{width}} | {right}")


def write_report(
    path: str,
    args: argparse.Namespace,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    results: list[Result],
) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "args": vars(args),
        "metadata": runtime_metadata(device),
        "data": {
            "train_tokens": int(len(train_data)),
            "val_tokens": int(len(val_data)),
            "total_tokens": int(len(train_data) + len(val_data)),
        },
        "seed_protocols": [
            {
                "model_seed": seed,
                "train_batch_seed": seed + 10_000,
                "eval_batch_seed": seed + 20_000,
                "sampling_seed": seed + 30_000,
            }
            for seed in sorted({result.seed for result in results})
        ],
        "summary": summarize_results(results),
        "pairwise_vs_constant": pairwise_comparisons(results),
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nreport: {report_path}")


def elapsed_train_seconds(start: float, eval_seconds: float, device: torch.device) -> float:
    synchronize_device(device)
    return max(time.perf_counter() - start - eval_seconds, 1e-9)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


def summarize_results(results: list[Result]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        grouped.setdefault(result.name, []).append(result)
    rows = []
    for name, group in sorted(grouped.items()):
        rows.append(
            {
                "name": name,
                "runs": len(group),
                "seeds": [result.seed for result in group],
                "average_width": group[0].average_width,
                "square_sum": group[0].square_sum,
                "target_square_sum": group[0].target_square_sum,
                "final_train_loss": mean(result.final_train_loss for result in group),
                "val_loss": mean(result.val_loss for result in group),
                "val_ppl": mean(result.val_ppl for result in group),
                "best_val_loss": mean(result.best_val_loss for result in group),
                "tokens_per_sec": mean(result.tokens_per_sec for result in group),
                "val_loss_mean": mean(result.val_loss for result in group),
                "val_loss_stdev": stdev(result.val_loss for result in group),
                "best_val_loss_mean": mean(result.best_val_loss for result in group),
                "best_val_loss_stdev": stdev(result.best_val_loss for result in group),
                "tokens_per_sec_mean": mean(result.tokens_per_sec for result in group),
                "tokens_per_sec_stdev": stdev(result.tokens_per_sec for result in group),
                "params": group[0].params,
                "widths": group[0].widths,
                "efficiency": group[0].efficiency,
            }
        )
    return rows


def pairwise_comparisons(results: list[Result]) -> list[dict[str, Any]]:
    by_seed_name = {(result.seed, result.name): result for result in results}
    names = sorted({result.name for result in results if result.name != "constant"})
    rows = []
    for name in names:
        deltas = []
        for result in results:
            if result.name != name:
                continue
            constant = by_seed_name.get((result.seed, "constant"))
            if constant is None:
                continue
            deltas.append(
                {
                    "val_loss_delta": result.val_loss - constant.val_loss,
                    "best_val_loss_delta": result.best_val_loss - constant.best_val_loss,
                    "tokens_per_sec_delta_pct": pct_delta(result.tokens_per_sec, constant.tokens_per_sec),
                    "val_loss_win": 1 if result.val_loss < constant.val_loss else 0,
                }
            )
        if deltas:
            rows.append(
                {
                    "name": name,
                    "runs": len(deltas),
                    "val_loss_delta_mean": mean(row["val_loss_delta"] for row in deltas),
                    "best_val_loss_delta_mean": mean(row["best_val_loss_delta"] for row in deltas),
                    "tokens_per_sec_delta_pct_mean": mean(row["tokens_per_sec_delta_pct"] for row in deltas),
                    "val_loss_wins": int(sum(row["val_loss_win"] for row in deltas)),
                }
            )
    return rows


def mean(values: Any) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else float("nan")


def stdev(values: Any) -> float:
    items = list(values)
    return float(statistics.stdev(items)) if len(items) > 1 else 0.0


def pct_delta(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


if __name__ == "__main__":
    main()
