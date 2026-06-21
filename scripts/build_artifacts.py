#!/usr/bin/env python3
"""Build blog and animation artifacts from a benchmark JSON report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_TITLE = "Constant vs Variable-Width Transformers"
CONSTANT_COLOR = "#2f6f9f"
VARIABLE_COLOR = "#c45a3c"
GRID_COLOR = "#d7dde3"
TEXT_COLOR = "#202833"
MUTED_COLOR = "#5d6875"
PANEL_FILL = "#f7f8fa"


def main() -> None:
    args = parse_args()
    outputs = build_artifacts(args.report, args.out_dir, args.title)
    print("wrote artifacts:")
    for path in outputs:
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="runs/last_run.json", help="Benchmark JSON report to read.")
    parser.add_argument("--out-dir", default="runs/artifacts", help="Directory for generated artifacts.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Title for the generated blog and HTML page.")
    return parser.parse_args()


def build_artifacts(report_path: Path, out_dir: Path, title: str = DEFAULT_TITLE) -> List[Path]:
    report_path = Path(report_path)
    out_dir = Path(out_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    constant, variable = load_pair(report)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        out_dir / "comparison_blog.md",
        out_dir / "width_schedule.svg",
        out_dir / "widths_animation.svg",
        out_dir / "loss_curve.svg",
        out_dir / "loss_animation.svg",
        out_dir / "comparison.html",
    ]

    outputs[0].write_text(render_blog(report, title, constant, variable), encoding="utf-8")
    outputs[1].write_text(render_width_svg(constant, variable, animated=False), encoding="utf-8")
    outputs[2].write_text(render_width_svg(constant, variable, animated=True), encoding="utf-8")
    outputs[3].write_text(render_loss_svg(constant, variable, animated=False), encoding="utf-8")
    outputs[4].write_text(render_loss_svg(constant, variable, animated=True), encoding="utf-8")
    outputs[5].write_text(render_html(report, title, constant, variable), encoding="utf-8")
    return outputs


def load_pair(report: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    results = report.get("results", [])
    by_name = {str(result.get("name", "")).lower(): result for result in results}
    try:
        return by_name["constant"], by_name["variable"]
    except KeyError as exc:
        names = ", ".join(sorted(name for name in by_name if name)) or "none"
        raise ValueError(f"report must contain constant and variable results; found {names}") from exc


def render_blog(
    report: Mapping[str, Any],
    title: str,
    constant: Mapping[str, Any],
    variable: Mapping[str, Any],
) -> str:
    args = report.get("args", {})
    data = report.get("data", {})
    seed_protocol = report.get("seed_protocol", {})
    created_at = str(report.get("created_at", "unknown"))
    steps = int(args.get("steps", latest_step(constant, variable)))
    total_tokens = int(data.get("total_tokens", 0))

    param_delta = pct_delta(number(variable, "params"), number(constant, "params"))
    val_delta = number(variable, "val_loss") - number(constant, "val_loss")
    ppl_delta = pct_delta(number(variable, "val_ppl"), number(constant, "val_ppl"))
    speed_delta = pct_delta(number(variable, "tokens_per_sec"), number(constant, "tokens_per_sec"))
    square_delta = pct_delta(number(variable, "square_sum"), number(constant, "square_sum"))

    if val_delta < 0:
        val_sentence = f"The variable-width model finished {abs(val_delta):.4f} validation-loss points lower."
    elif val_delta > 0:
        val_sentence = f"The variable-width model finished {val_delta:.4f} validation-loss points higher."
    else:
        val_sentence = "Both models finished with the same validation loss."

    return f"""# {title}

Generated from `{html_escape(str(report.get("args", {}).get("report_path", "runs/last_run.json")))}` at {created_at}.

This is a byte-level local benchmark, not a reproduction-scale language-model result. The bundled corpus is {total_tokens:,} bytes, the run uses {steps:,} training steps, and the comparison is most useful as a controlled local comparison of these two implementations.

## Experiment

- Constant model widths: `{format_widths(constant)}`
- Variable model widths: `{format_widths(variable)}`
- Seed protocol: model `{seed_protocol.get("model_seed", "n/a")}`, train batches `{seed_protocol.get("train_batch_seed", "n/a")}`, eval batches `{seed_protocol.get("eval_batch_seed", "n/a")}`, sampling `{seed_protocol.get("sampling_seed", "n/a")}`
- Training setup: layers `{args.get("layers", "n/a")}`, base width `{args.get("width", "n/a")}`, heads `{args.get("heads", "n/a")}`, block size `{args.get("block_size", "n/a")}`, batch size `{args.get("batch_size", "n/a")}`

## Results

| model | params | avg width | sum(width^2) | train loss | val loss | best val | perplexity | tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{metric_row(constant)}
{metric_row(variable)}

The variable-width run used {format_signed_pct(param_delta)} parameters, {format_signed_pct(square_delta)} `sum(width^2)`, and {format_signed_pct(speed_delta)} throughput relative to the constant baseline. {val_sentence} Its perplexity changed by {format_signed_pct(ppl_delta)}.

![Width schedule comparison](widths_animation.svg)

![Training-loss comparison](loss_animation.svg)

## Interpretation

The variable-width schedule spends capacity in the first and last blocks while narrowing the middle of the stack. That is the intended stress test: can the model keep endpoint representation capacity while saving work in intermediate layers? In this run, the validation metric and throughput should be read together with the parameter mismatch introduced by width quantization.

The honest research conclusion is conditional: this benchmark can reveal whether the implementation behaves sensibly, but it is not enough to claim a scaling result. For a stronger result, run larger corpora, multiple seeds, matched parameter/FLOP budgets, and confidence intervals.

## Samples

### Constant

```text
{constant.get("generation", "")}
```

### Variable

```text
{variable.get("generation", "")}
```
"""


def render_html(
    report: Mapping[str, Any],
    title: str,
    constant: Mapping[str, Any],
    variable: Mapping[str, Any],
) -> str:
    args = report.get("args", {})
    data = report.get("data", {})
    created_at = html_escape(str(report.get("created_at", "unknown")))
    total_tokens = int(data.get("total_tokens", 0))
    param_delta = format_signed_pct(pct_delta(number(variable, "params"), number(constant, "params")))
    val_delta = number(variable, "val_loss") - number(constant, "val_loss")
    speed_delta = format_signed_pct(pct_delta(number(variable, "tokens_per_sec"), number(constant, "tokens_per_sec")))
    val_copy = f"{val_delta:+.4f}"

    width_svg = render_width_svg(constant, variable, animated=True)
    loss_svg = render_loss_svg(constant, variable, animated=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202833;
      --muted: #5d6875;
      --rule: #d7dde3;
      --panel: #f7f8fa;
      --constant: {CONSTANT_COLOR};
      --variable: {VARIABLE_COLOR};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: grid;
      gap: 12px;
      border-bottom: 1px solid var(--rule);
      padding-bottom: 20px;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 4.5rem);
      line-height: 0.95;
      letter-spacing: 0;
      max-width: 980px;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 1.25rem;
      letter-spacing: 0;
    }}
    p {{ margin: 0; max-width: 820px; }}
    .muted {{ color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .stat {{
      border: 1px solid var(--rule);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-width: 0;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: clamp(1.25rem, 3vw, 2rem);
      overflow-wrap: anywhere;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .chart {{
      border: 1px solid var(--rule);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .chart svg {{ display: block; width: 100%; height: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--rule);
      text-align: right;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--rule);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-height: 160px;
      font-size: 0.9rem;
    }}
    @media (max-width: 820px) {{
      .stats, .chart-grid, .samples {{ grid-template-columns: 1fr; }}
      th, td {{ font-size: 0.86rem; padding: 8px 6px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html_escape(title)}</h1>
      <p class="muted">Generated {created_at} from a tiny byte-level benchmark with {total_tokens:,} total corpus bytes, {args.get("steps", "n/a")} training steps, and paired train/eval sampling seeds.</p>
    </header>

    <section class="stats" aria-label="headline metrics">
      <div class="stat"><span>Variable params</span><strong>{param_delta}</strong></div>
      <div class="stat"><span>Validation loss delta</span><strong>{val_copy}</strong></div>
      <div class="stat"><span>Throughput delta</span><strong>{speed_delta}</strong></div>
    </section>

    <section class="chart-grid" aria-label="animated comparisons">
      <div class="chart">{width_svg}</div>
      <div class="chart">{loss_svg}</div>
    </section>

    <h2>Metrics</h2>
    <table>
      <thead>
        <tr><th>model</th><th>params</th><th>avg width</th><th>sum(width^2)</th><th>train loss</th><th>val loss</th><th>best val</th><th>ppl</th><th>tok/s</th></tr>
      </thead>
      <tbody>
        {html_metric_row(constant)}
        {html_metric_row(variable)}
      </tbody>
    </table>

    <h2>Generated Samples</h2>
    <section class="samples">
      <pre><strong>constant</strong>

{html_escape(str(constant.get("generation", "")))}</pre>
      <pre><strong>variable</strong>

{html_escape(str(variable.get("generation", "")))}</pre>
    </section>
  </main>
</body>
</html>
"""


def render_width_svg(
    constant: Mapping[str, Any],
    variable: Mapping[str, Any],
    animated: bool,
) -> str:
    constant_widths = [int(width) for width in constant.get("widths", [])]
    variable_widths = [int(width) for width in variable.get("widths", [])]
    layer_count = max(len(constant_widths), len(variable_widths), 1)
    max_width = max(constant_widths + variable_widths + [1])
    width = 900
    height = 340
    left = 64
    top = 48
    chart_w = 780
    chart_h = 220
    baseline = top + chart_h
    group_w = chart_w / layer_count
    bar_w = min(26.0, group_w * 0.28)
    parts = [svg_open(width, height)]
    parts.append(svg_text(24, 28, "Layer width schedule", size=18, weight=700))
    parts.append(legend(width - 256, 25))
    parts.append(axis_lines(left, top, chart_w, chart_h, max_width, "width"))

    for idx in range(layer_count):
        cx = left + group_w * idx + group_w / 2
        c_width = constant_widths[idx] if idx < len(constant_widths) else 0
        v_width = variable_widths[idx] if idx < len(variable_widths) else 0
        parts.append(bar(cx - bar_w - 3, baseline, bar_w, chart_h * c_width / max_width, CONSTANT_COLOR, animated, idx))
        parts.append(bar(cx + 3, baseline, bar_w, chart_h * v_width / max_width, VARIABLE_COLOR, animated, idx + 0.35))
        parts.append(svg_text(cx, baseline + 28, str(idx + 1), anchor="middle", size=12, fill=MUTED_COLOR))
        if group_w >= 70:
            parts.append(svg_text(cx - bar_w / 2 - 3, baseline - chart_h * c_width / max_width - 6, str(c_width), anchor="middle", size=11, fill=CONSTANT_COLOR))
            parts.append(svg_text(cx + bar_w / 2 + 3, baseline - chart_h * v_width / max_width - 6, str(v_width), anchor="middle", size=11, fill=VARIABLE_COLOR))

    parts.append(svg_text(left + chart_w / 2, height - 16, "layer", anchor="middle", size=12, fill=MUTED_COLOR))
    parts.append("</svg>")
    return "\n".join(parts)


def render_loss_svg(
    constant: Mapping[str, Any],
    variable: Mapping[str, Any],
    animated: bool,
) -> str:
    constant_series = metric_series(constant, "train_loss")
    variable_series = metric_series(variable, "train_loss")
    all_points = constant_series + variable_series
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 0.0)]
        constant_series = list(all_points)
        variable_series = list(all_points)

    min_step = min(point[0] for point in all_points)
    max_step = max(point[0] for point in all_points)
    min_loss = min(point[1] for point in all_points)
    max_loss = max(point[1] for point in all_points)
    if math.isclose(min_loss, max_loss):
        min_loss -= 0.5
        max_loss += 0.5
    pad = (max_loss - min_loss) * 0.08
    min_loss -= pad
    max_loss += pad

    width = 900
    height = 340
    left = 66
    top = 48
    chart_w = 780
    chart_h = 220

    def project(point: Tuple[float, float]) -> Tuple[float, float]:
        step, loss = point
        x = left + ((step - min_step) / max(max_step - min_step, 1.0)) * chart_w
        y = top + (1.0 - ((loss - min_loss) / max(max_loss - min_loss, 1e-9))) * chart_h
        return x, y

    parts = [svg_open(width, height)]
    parts.append(svg_text(24, 28, "Training loss over time", size=18, weight=700))
    parts.append(legend(width - 256, 25))
    parts.append(line_axes(left, top, chart_w, chart_h, min_loss, max_loss, max_step))
    parts.append(loss_path(constant_series, project, CONSTANT_COLOR, animated, delay=0.0))
    parts.append(loss_path(variable_series, project, VARIABLE_COLOR, animated, delay=0.25))
    parts.append(svg_text(left + chart_w / 2, height - 16, "training step", anchor="middle", size=12, fill=MUTED_COLOR))
    parts.append("</svg>")
    return "\n".join(parts)


def metric_series(result: Mapping[str, Any], key: str) -> List[Tuple[float, float]]:
    raw_history = result.get("history") or []
    points = []
    for entry in raw_history:
        if key in entry and "step" in entry:
            points.append((float(entry["step"]), float(entry[key])))
    if not points and key == "train_loss" and "final_train_loss" in result:
        step = float(result.get("final_step", 1.0))
        points = [(0.0, float(result["final_train_loss"])), (step, float(result["final_train_loss"]))]
    elif len(points) == 1:
        step, value = points[0]
        points.insert(0, (0.0, value))
        if step == 0:
            points.append((1.0, value))
    return sorted(points)


def bar(x: float, baseline: float, width: float, height: float, fill: str, animated: bool, order: float) -> str:
    y = baseline - height
    if not animated:
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" rx="3" />'
    begin = 0.12 + order * 0.08
    return (
        f'<rect x="{x:.2f}" y="{baseline:.2f}" width="{width:.2f}" height="0" fill="{fill}" rx="3">'
        f'<animate attributeName="y" from="{baseline:.2f}" to="{y:.2f}" dur="0.65s" begin="{begin:.2f}s" fill="freeze" />'
        f'<animate attributeName="height" from="0" to="{height:.2f}" dur="0.65s" begin="{begin:.2f}s" fill="freeze" />'
        "</rect>"
    )


def loss_path(
    series: Sequence[Tuple[float, float]],
    project: Any,
    color: str,
    animated: bool,
    delay: float,
) -> str:
    if not series:
        return ""
    d = []
    for idx, point in enumerate(series):
        x, y = project(point)
        d.append(("M" if idx == 0 else "L") + f" {x:.2f} {y:.2f}")
    path = " ".join(d)
    circles = []
    for idx, point in enumerate(series):
        if idx % max(1, len(series) // 12) == 0 or idx == len(series) - 1:
            x, y = project(point)
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.82" />')
    if not animated:
        return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />' + "".join(circles)
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" '
        f'stroke-linecap="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">'
        f'<animate attributeName="stroke-dashoffset" from="1" to="0" dur="1.8s" begin="{delay:.2f}s" fill="freeze" />'
        "</path>"
        + f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="{delay + 1.35:.2f}s" fill="freeze" />'
        + "".join(circles)
        + "</g>"
    )


def axis_lines(left: float, top: float, chart_w: float, chart_h: float, max_value: float, label: str) -> str:
    baseline = top + chart_h
    parts = [
        f'<line x1="{left}" y1="{baseline}" x2="{left + chart_w}" y2="{baseline}" stroke="{GRID_COLOR}" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="{GRID_COLOR}" />',
    ]
    for tick in range(0, 5):
        value = max_value * tick / 4
        y = baseline - chart_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-dasharray="3 5" />')
        parts.append(svg_text(left - 10, y + 4, f"{value:.0f}", anchor="end", size=11, fill=MUTED_COLOR))
    parts.append(svg_text(16, top + chart_h / 2, label, anchor="middle", size=12, fill=MUTED_COLOR, rotate=-90))
    return "\n".join(parts)


def line_axes(
    left: float,
    top: float,
    chart_w: float,
    chart_h: float,
    min_loss: float,
    max_loss: float,
    max_step: float,
) -> str:
    baseline = top + chart_h
    parts = [
        f'<line x1="{left}" y1="{baseline}" x2="{left + chart_w}" y2="{baseline}" stroke="{GRID_COLOR}" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="{GRID_COLOR}" />',
    ]
    for tick in range(0, 5):
        value = min_loss + (max_loss - min_loss) * tick / 4
        y = baseline - chart_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-dasharray="3 5" />')
        parts.append(svg_text(left - 10, y + 4, f"{value:.2f}", anchor="end", size=11, fill=MUTED_COLOR))
    for tick in range(0, 5):
        value = max_step * tick / 4
        x = left + chart_w * tick / 4
        parts.append(svg_text(x, baseline + 22, f"{value:.0f}", anchor="middle", size=11, fill=MUTED_COLOR))
    parts.append(svg_text(16, top + chart_h / 2, "loss", anchor="middle", size=12, fill=MUTED_COLOR, rotate=-90))
    return "\n".join(parts)


def svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff" />'
    )


def legend(x: float, y: float) -> str:
    return (
        f'<rect x="{x}" y="{y - 13}" width="12" height="12" fill="{CONSTANT_COLOR}" rx="2" />'
        f'{svg_text(x + 18, y - 3, "constant", size=12, fill=TEXT_COLOR)}'
        f'<rect x="{x + 116}" y="{y - 13}" width="12" height="12" fill="{VARIABLE_COLOR}" rx="2" />'
        f'{svg_text(x + 134, y - 3, "variable", size=12, fill=TEXT_COLOR)}'
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    anchor: str = "start",
    size: int = 12,
    fill: str = TEXT_COLOR,
    weight: int = 400,
    rotate: Optional[int] = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        f'font-size="{size}" font-weight="{weight}"{transform}>{html_escape(text)}</text>'
    )


def metric_row(result: Mapping[str, Any]) -> str:
    return (
        f"| {result.get('name', '')} | {int(number(result, 'params')):,} | "
        f"{number(result, 'average_width'):.1f} | {int(number(result, 'square_sum')):,} | "
        f"{number(result, 'final_train_loss'):.4f} | {number(result, 'val_loss'):.4f} | "
        f"{number(result, 'best_val_loss', number(result, 'val_loss')):.4f} | "
        f"{number(result, 'val_ppl'):.2f} | {number(result, 'tokens_per_sec'):.0f} |"
    )


def html_metric_row(result: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_escape(str(result.get('name', '')))}</td>"
        f"<td>{int(number(result, 'params')):,}</td>"
        f"<td>{number(result, 'average_width'):.1f}</td>"
        f"<td>{int(number(result, 'square_sum')):,}</td>"
        f"<td>{number(result, 'final_train_loss'):.4f}</td>"
        f"<td>{number(result, 'val_loss'):.4f}</td>"
        f"<td>{number(result, 'best_val_loss', number(result, 'val_loss')):.4f}</td>"
        f"<td>{number(result, 'val_ppl'):.2f}</td>"
        f"<td>{number(result, 'tokens_per_sec'):.0f}</td>"
        "</tr>"
    )


def format_widths(result: Mapping[str, Any]) -> str:
    return "[" + ", ".join(str(int(width)) for width in result.get("widths", [])) + "]"


def latest_step(*results: Mapping[str, Any]) -> int:
    steps = []
    for result in results:
        for entry in result.get("history") or []:
            if "step" in entry:
                steps.append(int(float(entry["step"])))
    return max(steps) if steps else 0


def number(result: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = result.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct_delta(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def format_signed_pct(value: float) -> str:
    return f"{value:+.1f}%"


def html_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
