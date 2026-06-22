import math

from scripts.compare_paper_scale import build_comparison, render_markdown


def report(kind: str, loss: float, tokens_per_sec: float, params: int, avg_width: float) -> dict:
    return {
        "scale": {"name": "dense_200m"},
        "model": {
            "kind": kind,
            "params": params,
            "average_width": avg_width,
            "widths": [640, 640],
        },
        "metrics": {
            "smoothed_final_loss": loss,
            "tokens_per_sec": tokens_per_sec,
        },
        "data": {"tokens_seen": 10_003_415_040},
    }


def test_build_comparison_reports_primary_and_efficiency_deltas() -> None:
    constant = report("constant", loss=3.8, tokens_per_sec=300_000, params=200, avg_width=640.0)
    variable = report("variable", loss=3.7, tokens_per_sec=330_000, params=180, avg_width=580.0)

    comparison = build_comparison(constant, variable)

    assert comparison["scale"] == "dense_200m"
    assert math.isclose(comparison["deltas"]["smoothed_final_loss"], -0.1)
    assert comparison["deltas"]["tokens_per_sec_pct"] == 10.0
    assert comparison["deltas"]["params_pct"] == -10.0
    assert math.isclose(comparison["constant"]["perplexity"], math.exp(3.8))


def test_render_markdown_uses_loss_winner() -> None:
    constant = report("constant", loss=3.8, tokens_per_sec=300_000, params=200, avg_width=640.0)
    variable = report("variable", loss=3.7, tokens_per_sec=330_000, params=180, avg_width=580.0)

    markdown = render_markdown(build_comparison(constant, variable))

    assert "Winner by smoothed final loss: **variable**" in markdown
    assert "| variable | 180 | 580.0 | 3.7000 |" in markdown
