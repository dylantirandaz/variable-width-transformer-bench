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

This repo has two execution paths. The default path is a local byte-level
language-model benchmark so the two architectures can train and generate side
by side on a laptop. The paper-scale dense path uses cl100k token data,
4096-token sequences, torchrun/DDP, and Modal launchers for the public dense
configs, but it still expects you to provide the DCLM-scale data and GPU budget.

The local benchmark keeps the paper-facing defaults that make sense at this
scale:
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

The default corpus in `data/tiny_corpus.txt` is the full *White Nights* story
from Project Gutenberg's `White Nights and Other Stories` HTML edition:

```bash
make corpus
```

The extractor is [scripts/fetch_white_nights.py](scripts/fetch_white_nights.py).
It keeps the `WHITE NIGHTS` story section and stops before `NOTES FROM
UNDERGROUND`. The default loader unwraps single line breaks and preserves
blank-line paragraph breaks before byte tokenization. Use your own UTF-8 text
file with:

```bash
PYTHONPATH=src python -m vwt_bench.benchmark --data-path /path/to/text.txt
```

The default corpus is small, so bundled results should be treated as a local
comparison rather than a scaling claim. Use a larger corpus and multiple seeds
before making a performance claim.

The paper's published experiments use DCLM-scale data, 4096-token sequences,
200M-2B dense models, a 3B/1B-active MoE model, and multi-GPU training. The
paper-scale dense path below mirrors the dense configs and launch shape; it is
not a bundled replacement for the data or compute budget.

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

For the larger full-story run that is closer to the paper's smallest dense
architecture shape, use:

```bash
make modal-white-nights
```

That mode uses the full bundled story, CUDA bf16 autocast, 16 layers, width 640,
16 heads, 512-byte context, 1,000 training steps, and three replicated seeds.

The Modal launcher is [scripts/modal_benchmark.py](scripts/modal_benchmark.py).
It uses an A100 function by default, writes reports to a persistent Modal Volume
named `vwt-bench-runs`, and generates artifacts beside the report. Download a
report with Modal's volume CLI, for example:

```bash
.venv/bin/modal volume get vwt-bench-runs replicated.json runs/modal_replicated.json
```

## Paper-Scale Dense Runs

The paper's dense regime is a different code path from the tiny benchmark:

- tokenization: OpenAI `cl100k_base`-style IDs with vocab size `100263`;
- data: large DCLM-style text tokenized once into a flat `uint32` token file;
- sampling: sequential, no-repeat contiguous 4096-token sequences;
- training: CUDA bf16, RoPE, SwiGLU, bias-free projections, RMSNorm,
  μP-style attention scaling, AdamW `(0.9, 0.95)`, weight decay `0.1`;
- distributed launch: `torchrun`/DDP, with the official dense config batch
  schedules enforced by default.

The dense config table implemented in [src/vwt_bench/paper_scale.py](src/vwt_bench/paper_scale.py)
mirrors the authors' public configs. Training tokens are input tokens consumed
by optimizer steps; the stored memmap needs one extra terminal target token for
next-token prediction.

| scale | layers | width | seq len | tokens/step | steps | training tokens | memmap tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_200m` | 16 | 640 | 4096 | 524,288 | 19,080 | 10,003,415,040 | 10,003,415,041 |
| `dense_500m` | 24 | 960 | 4096 | 1,048,576 | 23,844 | 25,001,328,384 | 25,001,328,385 |
| `dense_1b` | 32 | 1280 | 4096 | 2,097,152 | 23,842 | 49,999,724,544 | 49,999,724,545 |
| `dense_2b` | 40 | 1600 | 4096 | 4,194,304 | 25,000 | 104,857,600,000 | 104,857,600,001 |

Install the scale dependencies with:

```bash
make setup-scale
```

Prepare a local cl100k token file from local text or JSONL shards:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_cl100k_dataset.py \
  --input '/path/to/dclm/**/*.jsonl' \
  --text-key text \
  --output data/dclm_cl100k_uint32.bin \
  --max-tokens 10003415041
```

For Hugging Face streaming datasets, use `--hf-dataset`, `--hf-config`, and
`--hf-split` instead of `--input`. `--max-tokens` is the stored memmap token
count, so use the table's `memmap tokens` value for full paper-scale runs. The
trainer validates that the file contains at least `steps * tokens_per_step + 1`
tokens. A real 10B-token run needs the resulting `.bin` file available on the
training node or in the Modal `vwt-paper-data` volume at
`/data/dclm_cl100k_uint32.bin`.

To build that file inside Modal from public DCLM-Baseline instead of uploading
it from a laptop:

```bash
make modal-prepare-dclm
```

That target defaults to `dense_200m` and DCLM-Baseline. Override the scale or
dataset without editing the Makefile:

```bash
make modal-prepare-dclm SCALE=dense_500m
make modal-prepare-dclm DCLM_DATASET=org/dataset-name
```

If an older volume already contains exactly the training-token count without
the terminal target token, append one EOS target token in place:

```bash
make modal-append-terminal-token SCALE=dense_200m
```

Run a single model on an 8-GPU node:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 scripts/train_paper_scale.py \
  --scale dense_200m \
  --model-kind variable \
  --train-bin /data/dclm_cl100k_uint32.bin
```

Run the 200M constant and variable pair on Modal 8xA100 after the data volume
contains `/data/dclm_cl100k_uint32.bin`:

```bash
make modal-paper-200m
```

For non-default paper-scale launches, use the generic target:

```bash
make modal-paper-scale SCALE=dense_200m MODEL_KIND=variable
make modal-paper-scale SCALE=dense_500m MODEL_KIND=both
```

The paper-scale trainer writes JSON reports under `runs/paper_scale` locally or
to the Modal `vwt-bench-runs` volume remotely. A Modal `--model-kind both` run
also writes `<scale>_comparison.md` with loss, perplexity, throughput, parameter,
and average-width deltas. The trainer refuses to silently change the official
effective token batch when launched with the wrong world size; pass
`--allow-batch-rescale` only for deliberate non-paper ablations.

Before interpreting the dense_200M run, use the preregistered analysis plan in
[docs/paper_scale_preregistration.md](docs/paper_scale_preregistration.md).
The blog draft in
[docs/blog_dense_200m_modal_draft.md](docs/blog_dense_200m_modal_draft.md)
is intentionally placeholder-based until both final reports exist.

Checkpoint behavior:

- `scripts/train_paper_scale.py` writes rank-0 checkpoints to
  `runs/paper_scale/checkpoints` by default every 1,000 steps and at the end.
- Checkpoints contain model weights, optimizer state, CLI args, and the global
  step. They do not store RNG state.
- `--resume /path/to/checkpoint.pt` loads the model and optimizer and continues
  from `checkpoint_step + 1`; sequential sampling derives the next data offset
  from that global step.
- `--keep-checkpoints 2` keeps the two newest checkpoints for the scale/model
  pair; use `--keep-checkpoints 0` to keep all.
- `scripts/modal_paper_scale.py` uses `/outputs/paper_scale_checkpoints` in the
  `vwt-bench-runs` volume, commits the volume after each printed checkpoint,
  resumes each model kind from the latest matching checkpoint by default, and
  skips a model kind when its final report already exists.
- Use `CHECKPOINT_INTERVAL=0`, `CHECKPOINT_AT_END=false`, `KEEP_CHECKPOINTS=0`,
  or `RESUME=false` on `make modal-paper-scale` when you deliberately want to
  change those defaults.

## Tests

```bash
make test
```

The tests cover the geometric width solver, paper-style residual expansion,
paper-scale dense configs, memmap token batching, chunked loss, and
forward/generation shape checks. They also cover paper-scale comparison math
and checkpoint-pruning behavior.
