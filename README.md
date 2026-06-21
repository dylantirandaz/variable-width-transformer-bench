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
- side-by-side generated text from the same prompt.

By default, the JSON report is written to `runs/last_run.json`.

The default corpus in `data/tiny_corpus.txt` is the *White Nights* text used by
this benchmark. Use your own UTF-8 text file with:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark --data-path /path/to/text.txt
```

The default corpus is small, so bundled results should be treated as a local
comparison rather than a scaling claim. Use a larger corpus and multiple seeds
before making a performance claim.

## Blog and Animation Artifacts

After running the benchmark, generate a Markdown blog draft, static SVG charts,
animated SVG charts, and a self-contained HTML comparison page:

```bash
make artifacts
```

Equivalent command:

```bash
PYTHONPATH=src python scripts/build_artifacts.py \
  --report runs/last_run.json \
  --out-dir runs/artifacts
```

Generated files:

- `runs/artifacts/comparison_blog.md`
- `runs/artifacts/width_schedule.svg`
- `runs/artifacts/widths_animation.svg`
- `runs/artifacts/loss_curve.svg`
- `runs/artifacts/loss_animation.svg`
- `runs/artifacts/comparison.html`

## Tests

```bash
make test
```

The tests cover the geometric width solver, paper-style residual expansion, and
forward/generation shape checks.
