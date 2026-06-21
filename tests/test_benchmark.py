import argparse

from vwt_bench.benchmark import learning_rate_for_step, width_quantum


def test_width_quantum_uses_even_rope_head_dimensions() -> None:
    assert width_quantum(heads=4, position_encoding="rope") == 8
    assert width_quantum(heads=4, position_encoding="learned") == 4


def test_learning_rate_warmup_and_decay() -> None:
    args = argparse.Namespace(
        disable_lr_schedule=False,
        lr=1.0,
        steps=100,
        warmup_fraction=0.1,
        lr_decay_power=1.0,
    )

    assert learning_rate_for_step(args, 1) == 0.1
    assert learning_rate_for_step(args, 10) == 1.0
    assert learning_rate_for_step(args, 100) == 0.0
