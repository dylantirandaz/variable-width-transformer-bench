"""Tiny decoder-only Transformers used by the benchmark."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resize_residual(
    x: torch.Tensor,
    out_dim: int,
    candidates: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Resize a residual stream using the paper's parameter-free rule.

    Shrinking truncates dimensions. Expanding copies each missing coordinate
    from the newest previous hidden state that has that coordinate, then pads
    with zeros if no previous state contains it.
    """

    in_dim = x.shape[-1]
    if in_dim == out_dim:
        return x
    if in_dim > out_dim:
        return x[..., :out_dim]

    fill_start = in_dim
    current = fill_start
    parts = []
    for candidate in candidates:
        candidate_dim = candidate.shape[-1]
        if candidate_dim <= current:
            continue
        end = min(candidate_dim, out_dim)
        parts.append(candidate[..., current:end])
        current = end
        if current >= out_dim:
            break

    needed = out_dim - fill_start
    if parts:
        expanded = torch.cat(parts, dim=-1)
        if expanded.shape[-1] < needed:
            pad = x.new_zeros(*x.shape[:-1], needed - expanded.shape[-1])
            expanded = torch.cat([expanded, pad], dim=-1)
    else:
        expanded = x.new_zeros(*x.shape[:-1], needed)
    return torch.cat([x, expanded], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        block_size: int,
        dropout: float = 0.0,
        position_encoding: str = "rope",
        rope_base: float = 10_000.0,
        attention_scale: str = "sqrt",
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        if position_encoding not in {"rope", "learned"}:
            raise ValueError("position_encoding must be 'rope' or 'learned'")
        if attention_scale not in {"sqrt", "mup"}:
            raise ValueError("attention_scale must be 'sqrt' or 'mup'")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.attention_scale = attention_scale
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, rope_base) if position_encoding == "rope" else None
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, steps, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)

        # SDPA selects FlashAttention/memory-efficient kernels on CUDA when
        # available. This is required for paper-scale 4096-token contexts.
        scale = 1.0 / self.head_dim if self.attention_scale == "mup" else None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=scale,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_drop(self.proj(y))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for per-layer attention head dimensions."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = q.shape[-2]
        positions = torch.arange(steps, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=q.device))
        cos = freqs.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype)[None, None, :, :]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = expansion * width
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def make_norm(kind: str, width: int, eps: float = 1e-5) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(width, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm(width, eps=eps)
    raise ValueError("norm must be 'layernorm' or 'rmsnorm'")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        block_size: int,
        dropout: float = 0.0,
        mlp_expansion: int = 4,
        position_encoding: str = "rope",
        rope_base: float = 10_000.0,
        norm: str = "layernorm",
        norm_eps: float = 1e-5,
        attention_scale: str = "sqrt",
    ) -> None:
        super().__init__()
        self.width = width
        self.ln_1 = make_norm(norm, width, norm_eps)
        self.attn = CausalSelfAttention(
            width,
            heads,
            block_size,
            dropout,
            position_encoding,
            rope_base,
            attention_scale,
        )
        self.ln_2 = make_norm(norm, width, norm_eps)
        self.mlp = SwiGLU(width, mlp_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Byte-level causal LM with configurable per-layer residual widths."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        base_width: int,
        widths: List[int],
        heads: int,
        dropout: float = 0.0,
        mlp_expansion: int = 4,
        position_encoding: str = "rope",
        rope_base: float = 10_000.0,
        init_std: float = 0.02,
        width_aware_init: bool = True,
        norm: str = "layernorm",
        norm_eps: float = 1e-5,
        attention_scale: str = "sqrt",
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must be non-empty")
        if position_encoding not in {"rope", "learned"}:
            raise ValueError("position_encoding must be 'rope' or 'learned'")
        if base_width % heads != 0:
            raise ValueError(f"base_width {base_width} must be divisible by heads {heads}")
        for width in widths:
            if width % heads != 0:
                raise ValueError(f"layer width {width} must be divisible by heads {heads}")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.base_width = base_width
        self.widths = list(widths)
        self.position_encoding = position_encoding
        self.init_std = init_std
        self.width_aware_init = width_aware_init
        self.norm = norm
        self.norm_eps = norm_eps
        self.attention_scale = attention_scale
        self.token_embedding = nn.Embedding(vocab_size, base_width)
        self.position_embedding = (
            nn.Embedding(block_size, base_width) if position_encoding == "learned" else None
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width=width,
                    heads=heads,
                    block_size=block_size,
                    dropout=dropout,
                    mlp_expansion=mlp_expansion,
                    position_encoding=position_encoding,
                    rope_base=rope_base,
                    norm=norm,
                    norm_eps=norm_eps,
                    attention_scale=attention_scale,
                )
                for width in widths
            ]
        )
        self.ln_f = make_norm(norm, widths[-1], norm_eps)
        self.lm_head = nn.Linear(base_width, vocab_size, bias=False)
        self._init_model_weights()

    def _init_module(self, module: nn.Module, std: float) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_model_weights(self) -> None:
        self._init_module(self.token_embedding, self.init_std)
        if self.position_embedding is not None:
            self._init_module(self.position_embedding, self.init_std)
        for width, block in zip(self.widths, self.blocks):
            std = self.init_std
            if self.width_aware_init:
                std *= math.sqrt(self.base_width / width)
            block.apply(lambda module, std=std: self._init_module(module, std))
        self._init_module(self.lm_head, self.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        loss_chunk_size: int = 0,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]] | tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, List[torch.Tensor]]]:
        batch, steps = input_ids.shape
        if steps > self.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.block_size}")

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(steps, device=input_ids.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        histories = [x]

        for width, block in zip(self.widths, self.blocks):
            x = resize_residual(x, width, reversed(histories))
            x = block(x)
            histories.append(x)

        x = self.ln_f(x)
        x = resize_residual(x, self.base_width, reversed(histories))

        loss = None
        logits = None
        if targets is not None and loss_chunk_size > 0 and not return_diagnostics:
            loss = self._chunked_cross_entropy(x, targets, loss_chunk_size)
        else:
            logits = self.lm_head(x)
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_diagnostics:
            return logits, loss, {"hidden_states": histories}
        return logits, loss

    def _chunked_cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        targets_flat = targets.reshape(-1)
        total_loss = None

        def loss_for_chunk(hidden_chunk: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
            logits = F.linear(hidden_chunk, self.lm_head.weight)
            return F.cross_entropy(logits, target_chunk, reduction="sum")

        use_checkpoint = hidden_flat.requires_grad and torch.is_grad_enabled()
        for start in range(0, hidden_flat.shape[0], chunk_size):
            hidden_chunk = hidden_flat[start : start + chunk_size]
            target_chunk = targets_flat[start : start + chunk_size]
            if use_checkpoint:
                chunk_loss = checkpoint(
                    loss_for_chunk,
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                chunk_loss = loss_for_chunk(hidden_chunk, target_chunk)
            total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss

        if total_loss is None:
            raise ValueError("targets must contain at least one token")
        return total_loss / targets_flat.numel()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
