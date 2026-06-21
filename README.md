# Variable-Width Transformer Bench

Small benchmark for comparing a constant-width decoder-only Transformer against
an X-shaped variable-width Transformer inspired by:

- Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy,
  Yoon Kim. "Variable-Width Transformers." arXiv:2606.18246.
  https://arxiv.org/abs/2606.18246

The paper's variable-width model keeps early and late layers wide, narrows the
middle layers, and uses parameter-free residual resizing:

- shrinking truncates residual dimensions;
- expanding restores each coordinate from the most recent earlier layer that
  actively processed that coordinate;
- missing coordinates are padded with zeros.

This repo implements a local byte-level language-model benchmark so the two
architectures can train and generate side by side on a laptop. It is not a
reproduction of the paper's DCLM-scale experiments.

The benchmark keeps the paper-facing defaults that make sense at this scale:
geometric X-shaped widths, the paper's `l* = 0.75L` and bottleneck ratio `0.3d`,
RoPE positional encoding, SwiGLU blocks, bias-free linear projections,
width-aware initialization, AdamW with `(0.9, 0.95)` betas, 0.1 weight decay,
8% warmup, and power learning-rate decay.

## Setup

```bash
cd /Users/dylantirandaz/variable-width-transformer-bench
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Benchmark Run

```bash
make benchmark
```

Equivalent command:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark \
  --steps 500 \
  --eval-iters 16 \
  --eval-interval 100 \
  --history-interval 5 \
  --layers 6 \
  --width 96 \
  --heads 4 \
  --batch-size 32 \
  --block-size 96 \
  --generate-tokens 240
```

The script prints:

- layer widths for both models;
- parameter counts;
- average layer width and `sum(width^2)` proxies;
- train throughput;
- validation loss/perplexity and best recorded validation loss;
- paired seed settings for model init, train batches, eval batches, and sampling;
- per-step history in the JSON report for plotting learning curves;
- scheduled learning rate in each history row;
- side-by-side generated text from the same prompt.

By default, the JSON report is written to `runs/last_run.json`.

For the main hypothesis test, run multiple seeds:

```bash
make replicated-benchmark
```

That writes `runs/replicated_benchmark.json` with per-seed results, aggregate
means/standard deviations, pairwise deltas against the constant-width baseline,
efficiency proxies, representation participation-ratio diagnostics, and
logit-lens losses.

For a paper-style schedule ablation, run:

```bash
make shape-sweep
```

That compares X, diamond, increasing, and decreasing geometric schedules against
the constant-width baseline.

The default corpus in `data/tiny_corpus.txt` is the *White Nights* text used by
this benchmark. It is soft-wrapped for readability; the default loader unwraps
single line breaks and preserves blank-line paragraph breaks before byte
tokenization. Use your own UTF-8 text file with:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark --data-path /path/to/text.txt
```

The default corpus is small, so bundled results should be treated as a local
comparison rather than a scaling claim. Use a larger corpus and multiple seeds
before making a performance claim.

The paper's published experiments use DCLM-scale data, 4096-token sequences,
200M-2B dense models, a 3B/1B-active MoE model, and multi-GPU training. This
repo follows the architecture and measurement methodology locally; it is not a
drop-in replacement for that compute budget.

## Modal GPU Runs

For larger GPU-backed runs without changing the benchmark code, use Modal:

```bash
make setup-modal
make modal-replicated
```

For a GPU shape sweep:

```bash
make modal-shape-sweep
```

The Modal launcher is [scripts/modal_benchmark.py](scripts/modal_benchmark.py).
It uses an A100 function by default, writes reports to a persistent Modal Volume
named `vwt-bench-runs`, and generates artifacts beside the report. Download a
report with Modal's volume CLI, for example:

```bash
.venv/bin/modal volume get vwt-bench-runs replicated.json runs/modal_replicated.json
```

## Tests

```bash
make test
```

The tests cover the geometric width solver, paper-style residual expansion, and
forward/generation shape checks.
