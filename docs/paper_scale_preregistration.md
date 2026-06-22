# Dense 200M Preregistration

Last update-on 2026-06-22, before any completed `dense_200m`
constant-vs-variable reports were available. 

## Question

Does the X-shaped variable-width Transformer beat the constant-width
Transformer in this repo when both are trained on the same `dense_200m` token
schedule?

## Fixed Setup

- Scale: `dense_200m`
- Data: public DCLM-Baseline text streamed and tokenized with `cl100k_base`
- Training tokens per model: `10,003,415,040`
- Stored memmap tokens: `10,003,415,041`, including one terminal target token
- Sequence length: `4096`
- Tokens per optimizer step: `524,288`
- Steps per model: `19,080`
- Launch shape: `torchrun --nproc_per_node=8` on Modal `A100:8`
- Precision: CUDA bf16 autocast
- Optimizer: AdamW, betas `(0.9, 0.95)`, eps `1e-10`, weight decay `0.1`
- Norm/attention: RMSNorm and muP-style attention scale
- Loss smoothing: final `1000` steps, sampled every `10` steps
- Variable-width schedule: geometric X shape, bottleneck layer ratio `0.75`,
  bottleneck width ratio `0.30`, quantized to 32

The run order is fixed: constant width first, then variable width. The completed
reports must be read from:

- `/outputs/dense_200m_constant.json`
- `/outputs/dense_200m_variable.json`
- `/outputs/dense_200m_comparison.md`

## Primary Metric

The primary metric is `metrics.smoothed_final_loss` from each JSON report. In
this trainer, that value is the mean of logged training losses
sampled every 10 steps over the final 1000 steps.

The decision rule is intentionally simple:

- The variable-width model wins if
  `variable.smoothed_final_loss < constant.smoothed_final_loss`.
- The constant-width model wins otherwise.

Partial logs, interrupted runs, and non-final checkpoints do not count.

## Secondary Metrics

These numbers are useful context, but they do not override the loss rule. They
may not be used to relabel the winner.

- Final perplexity: `exp(smoothed_final_loss)`
- Throughput: `metrics.tokens_per_sec`
- Parameter count: `model.params`
- Average layer width: `model.average_width`
- Loss delta per parameter delta
- Throughput-normalized loss comparison
- Wall-clock efficiency: tokens per second and elapsed seconds per completed
  10B-token run

## Inclusion Rules

A report only counts if it matches the Fixed Setup above. In particular, all of
these must be true:

- `data.tokens_seen == scale.total_tokens`
- `data.tokens_per_step == scale.tokens_per_step`
- `data.sequence_length == scale.sequence_length`
- `distributed.world_size == 8`
- `model.kind` matches the report file

Exclude anything in this list:

- Runs stopped before final report creation
- Runs with changed batch size do not count for the primary comparison; they
  may only be reported separately as non-paper ablations
- Runs using a different corpus, tokenizer, sequence length, precision, or
  optimizer
- Logs from failed infrastructure attempts

## Interpretation

This run tests one paper-scale dense configuration in this repo. It is not a
full reproduction of the paper's sweep: one scale, one dataset preparation
path, one seed, and one constant-vs-variable pair.

If the variable-width model wins, I will treat that as evidence that this
implementation reproduces the paper's direction at `dense_200m`. If the
constant-width model wins, I will treat that as a failed reproduction at this
scale, not as a proof that the paper is wrong.
