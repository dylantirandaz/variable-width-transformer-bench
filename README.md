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

This repo implements a tiny byte-level language-model benchmark so the two
architectures can train and generate side by side on a laptop. It is not a
reproduction of the paper's DCLM-scale experiments.

## Setup

```bash
cd /Users/dylantirandaz/variable-width-transformer-bench
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quick Run

```bash
make quick
```

Equivalent command:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark --steps 80 --eval-iters 8 --batch-size 16
```

The script prints:

- layer widths for both models;
- parameter counts;
- average layer width and `sum(width^2)` proxies;
- train throughput;
- validation loss/perplexity;
- side-by-side generated text from the same prompt.

## Larger Local Run

```bash
PYTHONPATH=src python -m vwt_bench.benchmark \
  --steps 500 \
  --layers 6 \
  --width 96 \
  --heads 4 \
  --batch-size 32 \
  --block-size 96 \
  --generate-tokens 240
```

Use your own UTF-8 text file with:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark --data-path /path/to/text.txt
```

## Tests

```bash
make test
```

The tests cover the geometric width solver, paper-style residual expansion, and
forward/generation shape checks.
