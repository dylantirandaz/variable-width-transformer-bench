from vwt_bench.paper_scale import get_paper_scale


def test_dense_200m_matches_official_token_schedule() -> None:
    scale = get_paper_scale("dense_200m")

    assert scale.layers == 16
    assert scale.width == 640
    assert scale.sequence_length == 4096
    assert scale.tokens_per_step == 524_288
    assert scale.sequences_per_step == 128
    assert scale.training_steps == 19_080
    assert scale.total_tokens == 10_003_415_040
    assert scale.micro_batch_size == 8
    assert scale.gradient_accumulation_steps == 2


def test_dense_2b_matches_official_token_schedule() -> None:
    scale = get_paper_scale("dense_2b")

    assert scale.layers == 40
    assert scale.width == 1600
    assert scale.tokens_per_step == 4_194_304
    assert scale.sequences_per_step == 1024
    assert scale.training_steps == 25_000
