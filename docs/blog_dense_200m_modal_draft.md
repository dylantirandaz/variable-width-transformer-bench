# Testing Variable-Width Transformers At 10B Tokens

The tiny local benchmark was useful for catching model bugs, but it was too
small to answer the actual question. I want to know whether the variable-width
idea still helps at the paper's small dense setting: `dense_200m`,
4096-token context, and about 10B training tokens.

I set up the comparison around one variable:

> Same data order, same token budget, same optimizer, same context length, same
> 8xA100 launch. The only intended difference is the layer-width schedule.

This is one run at one scale, not a full reproduction of the paper. The point
is narrower: replace a toy result with a 10B-token local ablation whose data
pipeline, model shape, and failure modes I can inspect directly.

## The Model Difference

The baseline is a normal decoder-only Transformer with width 640 in every
block.

The variable-width model keeps the same base width but changes residual-stream
width by layer. It uses an X-shaped schedule: wide early layers, narrow middle
layers, wide late layers. The resize rule has no learned projection:

- when a layer gets narrower, it drops the extra residual coordinates;
- when a later layer gets wider, each restored coordinate is copied from the
  most recent earlier hidden state that had that coordinate;
- if no earlier hidden state had that coordinate, it is filled with zero.

The experiment is not adding adapters or learned width projections. It is
testing whether uneven width allocation across depth is useful under this
training setup without learned resize projections.

## Run Setup

Both models use the same training setup.

| setting | value |
| --- | --- |
| scale | `dense_200m` |
| layers | 16 |
| base width | 640 |
| sequence length | 4096 |
| tokens per optimizer step | 524,288 |
| optimizer steps | 19,080 |
| training tokens per model | 10,003,415,040 |
| tokenizer | `cl100k_base` |
| data | public DCLM-Baseline stream |
| hardware | Modal `A100:8` |
| precision | CUDA bf16 autocast |
| optimizer | AdamW `(0.9, 0.95)`, eps `1e-10`, weight decay `0.1` |
| block details | RoPE, SwiGLU, RMSNorm, bias-free projections |

The variable run uses a geometric X schedule with bottleneck layer ratio
`0.75`, bottleneck width ratio `0.30`, and widths rounded to multiples of 32.

The data pipeline is fixed too. I stream DCLM-Baseline text, tokenize with
`cl100k_base`, and write one flat `uint32` memmap into a Modal volume. The
trainer reads contiguous 4096-token sequences in order, with no random batch
reuse in this path.

## Scoring

The primary metric is `metrics.smoothed_final_loss` from the completed JSON
reports. In this trainer, that is the mean of sampled training losses over the
final configured loss window.

The rule:

- lower variable-width smoothed loss means variable width wins;
- equal or higher variable-width smoothed loss means the constant-width
  baseline wins.

I will also look at perplexity, tokens/sec, parameter count, and average width.
Those are context for the result, not tie-breakers.

Partial logs do not count. Interrupted runs do not count. A checkpoint only
counts if it is resumed and the run reaches a final report.

## Results

This table should be filled only from completed reports with
`tokens_seen == 10,003,415,040`.

| model | params | avg width | smoothed final loss | perplexity | tokens/sec | tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | TBD | TBD | TBD | TBD | TBD | TBD |
| variable | TBD | TBD | TBD | TBD | TBD | TBD |

Primary comparison:

- Winner by smoothed final loss: TBD
- Variable loss delta: TBD
- Variable perplexity delta: TBD
- Variable throughput delta: TBD
- Variable parameter delta: TBD

## Notes From Getting It Running

The vocabulary loss computation was the first practical issue. With
`micro_batch=8`, `sequence_length=4096`, and `vocab=100263`, the full logits
tensor is large enough that the default cross-entropy path can run out of
memory on 40GB A100s. I changed the trainer to compute cross-entropy over token
chunks and checkpoint the chunk loss. The objective is unchanged; the trainer
just avoids holding the full logits allocation and its backward buffers at
once.

The Modal launch path was the second issue. A long run should not depend on a
local entrypoint sitting around and waiting for `.remote()`. If that local
input is canceled, Modal can cancel the training input too. The current launcher
starts the training as a spawned Modal function call and writes checkpoints
every 500 steps.

I keep only the two newest checkpoints for each model kind. These checkpoints
include model weights, optimizer state, CLI args, and the global step. Keeping
every checkpoint would waste a lot of volume space; keeping two is enough to
recover from an interrupted run without turning checkpoint storage into the
main cost.

## How I Will Read It

If the X-shaped run ends with lower smoothed loss, then this implementation has
evidence in the same direction as the paper's claim at `dense_200m`: depth-wise
width allocation can beat a uniform residual width under the same token budget.

If the constant run wins, that is also a real result. It would mean this setup
did not reproduce the gain at this scale. The next question would be whether
the miss comes from the shape schedule, the bottleneck ratio, initialization,
optimizer details, or just seed variance.

One run is still one run. I am not treating this as a universal statement about
variable-width Transformers. I am treating it as a 10B-token check: large
enough to expose scaling behavior that the toy corpus cannot show, but still
small enough that I can read the full training path.
