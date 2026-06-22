import argparse
from pathlib import Path

from scripts.train_paper_scale import prune_checkpoints, validate_runtime


def touch(path: Path) -> None:
    path.write_bytes(b"checkpoint")


def test_prune_checkpoints_keeps_newest_matching_files(tmp_path: Path) -> None:
    args = argparse.Namespace(
        scale="dense_200m",
        model_kind="constant",
        keep_checkpoints=2,
    )
    keep_1000 = tmp_path / "dense_200m_constant_step001000.pt"
    keep_1500 = tmp_path / "dense_200m_constant_step001500.pt"
    drop_500 = tmp_path / "dense_200m_constant_step000500.pt"
    other_kind = tmp_path / "dense_200m_variable_step000500.pt"
    unrelated = tmp_path / "dense_200m_constant_stepbad.pt"
    for path in [keep_1000, keep_1500, drop_500, other_kind, unrelated]:
        touch(path)

    prune_checkpoints(args, tmp_path)

    assert keep_1000.exists()
    assert keep_1500.exists()
    assert not drop_500.exists()
    assert other_kind.exists()
    assert unrelated.exists()


def test_validate_runtime_rejects_invalid_checkpoint_settings() -> None:
    args = argparse.Namespace(
        precision="bf16",
        device="cuda",
        tokens_per_step=8,
        sequence_length=4,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        sequences_per_step=2,
        allow_batch_rescale=False,
        steps=10,
        checkpoint_interval=-1,
        keep_checkpoints=2,
        loss_sample_interval=1,
    )

    try:
        validate_runtime(args, {"world_size": 2})
    except ValueError as exc:
        assert "--checkpoint-interval" in str(exc)
    else:
        raise AssertionError("expected invalid checkpoint interval to fail")


def test_validate_runtime_rejects_nonpositive_loss_sample_interval() -> None:
    args = argparse.Namespace(
        precision="bf16",
        device="cuda",
        tokens_per_step=8,
        sequence_length=4,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        sequences_per_step=2,
        allow_batch_rescale=False,
        steps=10,
        checkpoint_interval=0,
        keep_checkpoints=2,
        loss_sample_interval=0,
    )

    try:
        validate_runtime(args, {"world_size": 2})
    except ValueError as exc:
        assert "--loss-sample-interval" in str(exc)
    else:
        raise AssertionError("expected invalid loss sample interval to fail")


def test_validate_runtime_rejects_bf16_on_mps() -> None:
    args = argparse.Namespace(
        precision="bf16",
        device="mps",
        tokens_per_step=8,
        sequence_length=4,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        sequences_per_step=2,
        allow_batch_rescale=False,
        steps=10,
        checkpoint_interval=0,
        keep_checkpoints=2,
        loss_sample_interval=1,
    )

    try:
        validate_runtime(args, {"world_size": 1})
    except RuntimeError as exc:
        assert "bf16" in str(exc)
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("expected bf16 MPS run to fail")
