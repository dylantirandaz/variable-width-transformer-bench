import json
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


def test_build_artifacts_uses_aggregate_metrics_for_multiseed_reports(tmp_path: Path) -> None:
    builder = load_artifact_builder()
    report = tmp_path / "replicated.json"
    report.write_text(
        json.dumps(
            {
                "created_at": "2026-06-21T00:00:00Z",
                "args": {"steps": 2, "layers": 2, "width": 4, "heads": 1, "block_size": 8, "batch_size": 2},
                "data": {"total_tokens": 1000},
                "seed_protocols": [{"model_seed": 1}, {"model_seed": 2}],
                "summary": [
                    {"name": "constant", "runs": 2, "val_loss_mean": 3.0, "best_val_loss_mean": 2.5, "tokens_per_sec_mean": 200.0},
                    {"name": "variable", "runs": 2, "val_loss_mean": 2.0, "best_val_loss_mean": 1.5, "tokens_per_sec_mean": 300.0},
                ],
                "results": [
                    {
                        "name": "constant",
                        "seed": 1,
                        "widths": [4, 4],
                        "params": 10,
                        "average_width": 4.0,
                        "square_sum": 32,
                        "target_square_sum": 32,
                        "final_train_loss": 1.0,
                        "val_loss": 2.0,
                        "val_ppl": 10.0,
                        "best_val_loss": 2.0,
                        "tokens_per_sec": 100.0,
                        "history": [{"step": 1, "train_loss": 1.0}],
                    },
                    {
                        "name": "constant",
                        "seed": 2,
                        "widths": [4, 4],
                        "params": 10,
                        "average_width": 4.0,
                        "square_sum": 32,
                        "target_square_sum": 32,
                        "final_train_loss": 2.0,
                        "val_loss": 4.0,
                        "val_ppl": 30.0,
                        "best_val_loss": 3.0,
                        "tokens_per_sec": 300.0,
                    },
                    {
                        "name": "variable",
                        "seed": 1,
                        "widths": [6, 2],
                        "params": 12,
                        "average_width": 4.0,
                        "square_sum": 40,
                        "target_square_sum": 32,
                        "final_train_loss": 0.5,
                        "val_loss": 1.0,
                        "val_ppl": 5.0,
                        "best_val_loss": 1.0,
                        "tokens_per_sec": 250.0,
                        "history": [{"step": 1, "train_loss": 0.5}],
                    },
                    {
                        "name": "variable",
                        "seed": 2,
                        "widths": [6, 2],
                        "params": 12,
                        "average_width": 4.0,
                        "square_sum": 40,
                        "target_square_sum": 32,
                        "final_train_loss": 1.5,
                        "val_loss": 3.0,
                        "val_ppl": 15.0,
                        "best_val_loss": 2.0,
                        "tokens_per_sec": 350.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = builder.build_artifacts(report, tmp_path / "artifacts", "Replicated")
    blog = outputs[0].read_text(encoding="utf-8")

    assert "| constant | 10 | 4.0 | 32 | 1.5000 | 3.0000 | 2.5000 | 20.00 | 200 |" in blog
    assert "| variable | 12 | 4.0 | 40 | 1.0000 | 2.0000 | 1.5000 | 10.00 | 300 |" in blog


def test_build_artifacts_accepts_shape_sweep_variable_x_name(tmp_path: Path) -> None:
    builder = load_artifact_builder()
    report = tmp_path / "shape_sweep.json"
    report.write_text(
        json.dumps(
            {
                "created_at": "2026-06-21T00:00:00Z",
                "args": {"steps": 1, "layers": 2, "width": 4, "heads": 1, "block_size": 8, "batch_size": 2},
                "data": {"total_tokens": 1000},
                "results": [
                    {
                        "name": "constant",
                        "widths": [4, 4],
                        "params": 10,
                        "average_width": 4.0,
                        "square_sum": 32,
                        "target_square_sum": 32,
                        "final_train_loss": 1.0,
                        "val_loss": 3.0,
                        "val_ppl": 20.0,
                        "best_val_loss": 3.0,
                        "tokens_per_sec": 200.0,
                        "history": [{"step": 1, "train_loss": 1.0}],
                    },
                    {
                        "name": "variable_x",
                        "widths": [6, 2],
                        "params": 12,
                        "average_width": 4.0,
                        "square_sum": 40,
                        "target_square_sum": 32,
                        "final_train_loss": 1.0,
                        "val_loss": 2.0,
                        "val_ppl": 10.0,
                        "best_val_loss": 2.0,
                        "tokens_per_sec": 180.0,
                        "history": [{"step": 1, "train_loss": 1.0}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = builder.build_artifacts(report, tmp_path / "artifacts", "Shape Sweep")
    blog = outputs[0].read_text(encoding="utf-8")

    assert "| variable_x | 12 | 4.0 | 40 | 1.0000 | 2.0000 | 2.0000 | 10.00 | 180 |" in blog
