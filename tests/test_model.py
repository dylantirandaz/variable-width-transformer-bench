import torch

from vwt_bench.model import TinyTransformerLM, resize_residual


def test_resize_residual_restores_from_newest_candidate() -> None:
    current = torch.tensor([[[1.0, 2.0]]])
    older = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    newest = torch.tensor([[[100.0, 200.0, 300.0]]])

    out = resize_residual(current, 5, candidates=[newest, older])

    assert out.shape[-1] == 5
    assert torch.equal(out[..., :2], current)
    assert out[..., 2].item() == 300.0
    assert out[..., 3].item() == 40.0
    assert out[..., 4].item() == 0.0


def test_tiny_transformer_forward_and_generate_shapes() -> None:
    torch.manual_seed(123)
    model = TinyTransformerLM(
        vocab_size=32,
        block_size=8,
        base_width=16,
        widths=[24, 16, 24],
        heads=4,
    )
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    out = model.generate(x[:, :2], max_new_tokens=3, temperature=1.0, top_k=8)
    assert out.shape == (2, 5)


def test_tiny_transformer_supports_paper_norm_and_attention_scale() -> None:
    torch.manual_seed(123)
    model = TinyTransformerLM(
        vocab_size=32,
        block_size=8,
        base_width=16,
        widths=[32, 16, 32],
        heads=4,
        norm="rmsnorm",
        attention_scale="mup",
        init_std=0.1,
    )
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None


def test_chunked_loss_matches_full_logits_loss() -> None:
    torch.manual_seed(123)
    model = TinyTransformerLM(
        vocab_size=32,
        block_size=8,
        base_width=16,
        widths=[24, 16, 24],
        heads=4,
    )
    x = torch.randint(0, 32, (2, 8))
    _, full_loss = model(x, x)
    logits, chunked_loss = model(x, x, loss_chunk_size=3)

    assert logits is None
    assert full_loss is not None
    assert chunked_loss is not None
    assert torch.allclose(chunked_loss, full_loss, atol=1e-6)


def test_learned_position_encoding_allows_odd_head_dim() -> None:
    model = TinyTransformerLM(
        vocab_size=32,
        block_size=8,
        base_width=16,
        widths=[24, 12, 24],
        heads=4,
        position_encoding="learned",
    )
    x = torch.randint(0, 32, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, 32)


def test_rope_requires_even_head_dim() -> None:
    try:
        TinyTransformerLM(
            vocab_size=32,
            block_size=8,
            base_width=16,
            widths=[24, 12, 24],
            heads=4,
        )
    except ValueError as exc:
        assert "RoPE requires an even head_dim" in str(exc)
    else:
        raise AssertionError("expected odd RoPE head dimension to fail")
