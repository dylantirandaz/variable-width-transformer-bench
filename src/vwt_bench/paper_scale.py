"""Paper-scale dense training configurations.

The values mirror the public dense configs released with arXiv:2606.18246.
Batch size is represented as tokens per optimizer step, matching the configs'
``c`` scheduler value rather than the ambiguous table shorthand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CL100K_PAPER_VOCAB_SIZE = 100_263
CL100K_EOS_TOKEN = 100_261
CL100K_PAD_TOKEN = 100_262


@dataclass(frozen=True)
class PaperScaleConfig:
    name: str
    nominal_params: str
    layers: int
    width: int
    heads: int
    sequence_length: int
    vocab_size: int
    tokens_per_step: int
    training_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    decay_steps: int
    bottleneck_layer_ratio: float
    bottleneck_width_ratio: float
    quantize_to: int
    init_std: float
    norm: str
    attention_scale: str

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.training_steps

    @property
    def sequences_per_step(self) -> int:
        if self.tokens_per_step % self.sequence_length != 0:
            raise ValueError("tokens_per_step must divide evenly by sequence_length")
        return self.tokens_per_step // self.sequence_length

    def steps_for_tokens(self, tokens: int) -> int:
        return math.ceil(tokens / self.tokens_per_step)


PAPER_DENSE_SCALES: dict[str, PaperScaleConfig] = {
    "dense_200m": PaperScaleConfig(
        name="dense_200m",
        nominal_params="200M",
        layers=16,
        width=640,
        heads=16,
        sequence_length=4096,
        vocab_size=CL100K_PAPER_VOCAB_SIZE,
        tokens_per_step=524_288,
        training_steps=19_080,
        micro_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=1_526,
        decay_steps=17_554,
        bottleneck_layer_ratio=0.75,
        bottleneck_width_ratio=0.30,
        quantize_to=32,
        init_std=0.1,
        norm="rmsnorm",
        attention_scale="mup",
    ),
    "dense_500m": PaperScaleConfig(
        name="dense_500m",
        nominal_params="500M",
        layers=24,
        width=960,
        heads=16,
        sequence_length=4096,
        vocab_size=CL100K_PAPER_VOCAB_SIZE,
        tokens_per_step=1_048_576,
        training_steps=23_844,
        micro_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=1_908,
        decay_steps=21_936,
        bottleneck_layer_ratio=0.75,
        bottleneck_width_ratio=0.30,
        quantize_to=32,
        init_std=0.1,
        norm="rmsnorm",
        attention_scale="mup",
    ),
    "dense_1b": PaperScaleConfig(
        name="dense_1b",
        nominal_params="1B",
        layers=32,
        width=1280,
        heads=16,
        sequence_length=4096,
        vocab_size=CL100K_PAPER_VOCAB_SIZE,
        tokens_per_step=2_097_152,
        training_steps=23_842,
        micro_batch_size=2,
        gradient_accumulation_steps=32,
        warmup_steps=1_907,
        decay_steps=21_935,
        bottleneck_layer_ratio=0.75,
        bottleneck_width_ratio=0.30,
        quantize_to=32,
        init_std=0.1,
        norm="rmsnorm",
        attention_scale="mup",
    ),
    "dense_2b": PaperScaleConfig(
        name="dense_2b",
        nominal_params="2B",
        layers=40,
        width=1600,
        heads=16,
        sequence_length=4096,
        vocab_size=CL100K_PAPER_VOCAB_SIZE,
        tokens_per_step=4_194_304,
        training_steps=25_000,
        micro_batch_size=1,
        gradient_accumulation_steps=128,
        warmup_steps=2_000,
        decay_steps=23_000,
        bottleneck_layer_ratio=0.75,
        bottleneck_width_ratio=0.30,
        quantize_to=32,
        init_std=0.1,
        norm="rmsnorm",
        attention_scale="mup",
    ),
}


def get_paper_scale(name: str) -> PaperScaleConfig:
    try:
        return PAPER_DENSE_SCALES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PAPER_DENSE_SCALES))
        raise ValueError(f"unknown paper scale {name!r}; expected one of: {allowed}") from exc
