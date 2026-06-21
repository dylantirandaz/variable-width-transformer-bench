"""Width schedules for the tiny benchmark.

The X-shaped schedule follows the derivation in arXiv:2606.18246 at benchmark
scale: geometric shrink to a bottleneck, geometric expansion back to an equal
endpoint width, and approximate parameter matching through the sum of squared
layer widths plus the paper's endpoint correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


@dataclass(frozen=True)
class WidthSchedule:
    widths: List[int]
    bottleneck_layer: int
    alpha_down: float
    alpha_up: float
    target_bottleneck: float
    average_width: float
    square_sum: int
    target_square_sum: int


def uniform_widths(num_layers: int, width: int) -> WidthSchedule:
    widths = [width] * num_layers
    return WidthSchedule(
        widths=widths,
        bottleneck_layer=num_layers,
        alpha_down=1.0,
        alpha_up=1.0,
        target_bottleneck=float(width),
        average_width=float(width),
        square_sum=sum(w * w for w in widths),
        target_square_sum=num_layers * width * width,
    )


def x_shape_widths(
    num_layers: int,
    base_width: int,
    bottleneck_layer_ratio: float = 0.75,
    bottleneck_width_ratio: float = 0.30,
    quantize_to: int = 1,
    mlp_expansion: int = 4,
    endpoint_correction: bool = True,
) -> WidthSchedule:
    """Build an X-shaped, roughly parameter-matched geometric width schedule.

    Args:
        num_layers: Number of Transformer blocks.
        base_width: Constant-width baseline hidden size.
        bottleneck_layer_ratio: Layer index ratio for l*. The result is
            floored, matching the paper's ratio-style parameterization while
            keeping tiny layer counts usable.
        bottleneck_width_ratio: Target d_l* / base_width.
        quantize_to: Round each width to this multiple, usually the head count.
        mlp_expansion: SwiGLU intermediate width multiplier.
        endpoint_correction: Include the paper's unused endpoint parameter
            correction for first attention QKV and final MLP down projection.
    """

    if num_layers < 2:
        raise ValueError("x_shape_widths requires at least two layers")
    if base_width <= 0:
        raise ValueError("base_width must be positive")
    if not 0.0 < bottleneck_width_ratio <= 1.0:
        raise ValueError("bottleneck_width_ratio must be in (0, 1]")
    if not 0.0 < bottleneck_layer_ratio <= 1.0:
        raise ValueError("bottleneck_layer_ratio must be in (0, 1]")
    if quantize_to <= 0:
        raise ValueError("quantize_to must be positive")

    bottleneck_layer = int(num_layers * bottleneck_layer_ratio)
    bottleneck_layer = min(max(1, bottleneck_layer), num_layers - 1)
    target_bottleneck = max(float(quantize_to), base_width * bottleneck_width_ratio)

    def factors(alpha_down: float) -> List[float]:
        if bottleneck_layer == num_layers:
            alpha_up = 1.0
        else:
            alpha_up = alpha_down ** (-(bottleneck_layer - 1) / (num_layers - bottleneck_layer))
        out = []
        for layer in range(1, num_layers + 1):
            if layer <= bottleneck_layer:
                out.append(alpha_down ** (layer - 1))
            else:
                out.append((alpha_down ** (bottleneck_layer - 1)) * (alpha_up ** (layer - bottleneck_layer)))
        return out

    def alphas(alpha_down: float) -> tuple[float, float]:
        if bottleneck_layer == num_layers:
            return alpha_down, 1.0
        return alpha_down, alpha_down ** (-(bottleneck_layer - 1) / (num_layers - bottleneck_layer))

    def endpoint_width(alpha_down: float) -> float:
        width_factors = factors(alpha_down)
        s2 = sum(c * c for c in width_factors)
        if not endpoint_correction:
            return base_width * math.sqrt(num_layers / s2)

        # K = attention projections + SwiGLU projections.
        # Attention contributes Q, K, V, and output projections: 4 d^2.
        # SwiGLU contributes gate, up, and down projections: 3 E d^2.
        k = 4 + 3 * mlp_expansion
        correction = 3 + mlp_expansion

        roots = []
        for indicator in (0, 1):
            a = k * s2 - indicator * correction
            b = indicator * base_width * correction
            c = -num_layers * k * base_width * base_width
            disc = b * b - 4 * a * c
            if a <= 0 or disc < 0:
                continue
            root = (-b + math.sqrt(disc)) / (2 * a)
            is_wider = root > base_width + 1e-9
            if bool(indicator) == is_wider or abs(root - base_width) <= 1e-9:
                roots.append(root)
        if not roots:
            return base_width * math.sqrt(num_layers / s2)
        return max(roots)

    def bottleneck_for(alpha_down: float) -> float:
        return endpoint_width(alpha_down) * (alpha_down ** (bottleneck_layer - 1))

    if target_bottleneck >= base_width:
        alpha_down = 1.0
    else:
        lo, hi = 1e-5, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if bottleneck_for(mid) < target_bottleneck:
                lo = mid
            else:
                hi = mid
        alpha_down = hi

    alpha_down, alpha_up = alphas(alpha_down)
    endpoint = endpoint_width(alpha_down)
    raw_widths = [endpoint * c for c in factors(alpha_down)]
    widths = [_round_to_multiple(w, quantize_to) for w in raw_widths]
    widths[-1] = widths[0]

    return WidthSchedule(
        widths=widths,
        bottleneck_layer=bottleneck_layer,
        alpha_down=alpha_down,
        alpha_up=alpha_up,
        target_bottleneck=target_bottleneck,
        average_width=sum(widths) / len(widths),
        square_sum=sum(w * w for w in widths),
        target_square_sum=num_layers * base_width * base_width,
    )


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)
