#!/usr/bin/env python3
"""Compare constant and variable paper-scale training reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = parse_args()
    constant = json.loads(Path(args.constant).read_text(encoding="utf-8"))
    variable = json.loads(Path(args.variable).read_text(encoding="utf-8"))
    comparison = build_comparison(constant, variable)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".md":
        out_path.write_text(render_markdown(comparison), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"comparison: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def build_comparison(constant: dict, variable: dict) -> dict:
    c_loss = constant["metrics"]["smoothed_final_loss"]
    v_loss = variable["metrics"]["smoothed_final_loss"]
    c_tps = constant["metrics"]["tokens_per_sec"]
    v_tps = variable["metrics"]["tokens_per_sec"]
    return {
        "scale": constant["scale"]["name"],
        "constant": summarize(constant),
        "variable": summarize(variable),
        "deltas": {
            "smoothed_final_loss": v_loss - c_loss,
            "perplexity_pct": pct_delta(math.exp(v_loss), math.exp(c_loss)),
            "tokens_per_sec_pct": pct_delta(v_tps, c_tps),
            "params_pct": pct_delta(variable["model"]["params"], constant["model"]["params"]),
            "average_width_pct": pct_delta(
                variable["model"]["average_width"],
                constant["model"]["average_width"],
            ),
        },
    }


def summarize(report: dict) -> dict:
    return {
        "params": report["model"]["params"],
        "average_width": report["model"]["average_width"],
        "smoothed_final_loss": report["metrics"]["smoothed_final_loss"],
        "perplexity": math.exp(report["metrics"]["smoothed_final_loss"]),
        "tokens_per_sec": report["metrics"]["tokens_per_sec"],
        "tokens_seen": report["data"]["tokens_seen"],
        "widths": report["model"]["widths"],
    }


def pct_delta(new_value: float, old_value: float) -> float:
    return ((new_value - old_value) / old_value) * 100.0 if old_value else 0.0


def render_markdown(comparison: dict) -> str:
    c = comparison["constant"]
    v = comparison["variable"]
    d = comparison["deltas"]
    winner = "variable" if d["smoothed_final_loss"] < 0 else "constant"
    return "\n".join(
        [
            f"# Paper-Scale {comparison['scale']} Comparison",
            "",
            "| model | params | avg width | final loss | perplexity | tokens/sec | tokens seen |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            row("constant", c),
            row("variable", v),
            "",
            f"Winner by smoothed final loss: **{winner}**.",
            "",
            f"- Variable loss delta: `{d['smoothed_final_loss']:.6f}`",
            f"- Variable perplexity delta: `{d['perplexity_pct']:.2f}%`",
            f"- Variable throughput delta: `{d['tokens_per_sec_pct']:.2f}%`",
            f"- Variable parameter delta: `{d['params_pct']:.2f}%`",
            f"- Variable average-width delta: `{d['average_width_pct']:.2f}%`",
            "",
        ]
    )


def row(name: str, report: dict) -> str:
    return (
        f"| {name} | {report['params']:,} | {report['average_width']:.1f} | "
        f"{report['smoothed_final_loss']:.4f} | {report['perplexity']:.2f} | "
        f"{report['tokens_per_sec']:.0f} | {report['tokens_seen']:,} |"
    )


if __name__ == "__main__":
    main()
