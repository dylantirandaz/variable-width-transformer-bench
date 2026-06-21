import importlib.util
from pathlib import Path


def load_artifact_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_artifacts.py"
    spec = importlib.util.spec_from_file_location("build_artifacts", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_artifacts_from_report(tmp_path: Path) -> None:
    builder = load_artifact_builder()
    report = tmp_path / "run.json"
    report.write_text(
        """
{
  "created_at": "2026-06-21T00:00:00Z",
  "args": {
    "report_path": "runs/example.json",
    "steps": 500,
    "layers": 6,
    "width": 96,
    "heads": 4,
    "block_size": 96,
    "batch_size": 32
  },
  "data": {"train_tokens": 3237, "val_tokens": 360, "total_tokens": 3597},
  "seed_protocol": {
    "model_seed": 1337,
    "train_batch_seed": 11337,
    "eval_batch_seed": 21337,
    "sampling_seed": 31337
  },
  "results": [
    {
      "name": "constant",
      "widths": [96, 96, 96, 96, 96, 96],
      "params": 942528,
      "average_width": 96.0,
      "square_sum": 55296,
      "target_square_sum": 55296,
      "final_train_loss": 1.9,
      "val_loss": 3.2,
      "val_ppl": 24.5,
      "best_val_loss": 3.2,
      "tokens_per_sec": 48000,
      "history": [
        {"step": 100, "train_loss": 4.2, "val_loss": 4.8, "tokens_per_sec": 44000},
        {"step": 200, "train_loss": 3.4, "val_loss": 4.1, "tokens_per_sec": 46000},
        {"step": 300, "train_loss": 2.7, "val_loss": 3.7, "tokens_per_sec": 47000},
        {"step": 400, "train_loss": 2.2, "val_loss": 3.4, "tokens_per_sec": 47500},
        {"step": 500, "train_loss": 1.9, "val_loss": 3.2, "tokens_per_sec": 48000}
      ],
      "generation": "constant sample"
    },
    {
      "name": "variable",
      "widths": [148, 84, 48, 28, 64, 148],
      "params": 986440,
      "average_width": 86.7,
      "square_sum": 58048,
      "target_square_sum": 55296,
      "final_train_loss": 1.8,
      "val_loss": 3.1,
      "val_ppl": 22.2,
      "best_val_loss": 3.1,
      "tokens_per_sec": 30000,
      "history": [
        {"step": 100, "train_loss": 4.1, "val_loss": 4.7, "tokens_per_sec": 28500},
        {"step": 200, "train_loss": 3.2, "val_loss": 4.0, "tokens_per_sec": 29200},
        {"step": 300, "train_loss": 2.5, "val_loss": 3.6, "tokens_per_sec": 29600},
        {"step": 400, "train_loss": 2.0, "val_loss": 3.3, "tokens_per_sec": 29800},
        {"step": 500, "train_loss": 1.8, "val_loss": 3.1, "tokens_per_sec": 30000}
      ],
      "generation": "variable sample"
    }
  ]
}
""",
        encoding="utf-8",
    )

    outputs = builder.build_artifacts(report, tmp_path / "artifacts", "Example")

    assert len(outputs) == 6
    for output in outputs:
        assert output.exists()
    assert "byte-level local benchmark" in outputs[0].read_text(encoding="utf-8")
    assert "<svg" in outputs[2].read_text(encoding="utf-8")
    assert "variable sample" in outputs[5].read_text(encoding="utf-8")
