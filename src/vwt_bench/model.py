"""Tiny decoder-only Transformers used by the benchmark."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, width: int, heads: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, steps, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.mask[:, :, :steps, :steps], torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_drop(self.proj(y))


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        block_size: int,
        dropout: float = 0.0,
        mlp_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.width = width
        self.ln_1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(width)
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
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must be non-empty")
        if base_width % heads != 0:
            raise ValueError(f"base_width {base_width} must be divisible by heads {heads}")
        for width in widths:
            if width % heads != 0:
                raise ValueError(f"layer width {width} must be divisible by heads {heads}")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.base_width = base_width
        self.widths = list(widths)
        self.token_embedding = nn.Embedding(vocab_size, base_width)
        self.position_embedding = nn.Embedding(block_size, base_width)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width=width,
                    heads=heads,
                    block_size=block_size,
                    dropout=dropout,
                    mlp_expansion=mlp_expansion,
                )
                for width in widths
            ]
        )
        self.ln_f = nn.LayerNorm(widths[-1])
        self.lm_head = nn.Linear(base_width, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, steps = input_ids.shape
        if steps > self.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.block_size}")

        positions = torch.arange(steps, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        histories = [x]

        for width, block in zip(self.widths, self.blocks):
            x = resize_residual(x, width, reversed(histories))
            x = block(x)
            histories.append(x)

        x = self.ln_f(x)
        x = resize_residual(x, self.base_width, reversed(histories))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

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
