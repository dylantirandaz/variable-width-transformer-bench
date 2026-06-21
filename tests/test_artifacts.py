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
    "steps": 2,
    "layers": 3,
    "width": 32,
    "heads": 4,
    "block_size": 16,
    "batch_size": 2
  },
  "data": {"train_tokens": 90, "val_tokens": 10, "total_tokens": 100},
  "seed_protocol": {
    "model_seed": 1,
    "train_batch_seed": 10001,
    "eval_batch_seed": 20001,
    "sampling_seed": 30001
  },
  "results": [
    {
      "name": "constant",
      "widths": [32, 32, 32],
      "params": 1000,
      "average_width": 32.0,
      "square_sum": 3072,
      "target_square_sum": 3072,
      "final_train_loss": 4.0,
      "val_loss": 4.2,
      "val_ppl": 66.7,
      "best_val_loss": 4.2,
      "tokens_per_sec": 1000,
      "history": [
        {"step": 1, "train_loss": 5.0, "tokens_per_sec": 900},
        {"step": 2, "train_loss": 4.0, "val_loss": 4.2, "tokens_per_sec": 1000}
      ],
      "generation": "constant sample"
    },
    {
      "name": "variable",
      "widths": [40, 8, 40],
      "params": 1050,
      "average_width": 29.3,
      "square_sum": 3264,
      "target_square_sum": 3072,
      "final_train_loss": 3.8,
      "val_loss": 4.1,
      "val_ppl": 60.3,
      "best_val_loss": 4.1,
      "tokens_per_sec": 1200,
      "history": [
        {"step": 1, "train_loss": 5.1, "tokens_per_sec": 1000},
        {"step": 2, "train_loss": 3.8, "val_loss": 4.1, "tokens_per_sec": 1200}
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
