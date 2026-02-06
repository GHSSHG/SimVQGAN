from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


def _dropout(x: jnp.ndarray, *, rate: float, train: bool, rng: jax.Array | None) -> jnp.ndarray:
    if (not train) or rate <= 0.0:
        return x
    if rng is None:
        raise ValueError("dropout requires rng when train=True")
    keep_prob = 1.0 - rate
    keep_mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return jnp.where(keep_mask, x / keep_prob, 0.0)


class TransformerBlock1D(nn.Module):
    """Pre-norm Transformer block for sequence latents shaped (B, T, C)."""

    dim: int
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        hidden_dim = int(round(self.dim * float(self.mlp_ratio)))
        if hidden_dim <= 0:
            raise ValueError(f"mlp hidden dim must be positive, got {hidden_dim}")

        self.norm1 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=float(self.dropout),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attn",
        )
        self.norm2 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm2")
        self.ffn_in = nn.Dense(
            hidden_dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ffn_in",
        )
        self.ffn_out = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ffn_out",
        )

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"TransformerBlock1D expects (B,T,C), got {x.shape}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dim {x.shape[-1]} != block dim {self.dim}")

        deterministic = (not train) or (self.dropout <= 0.0)
        attn_rng = ffn_rng = resid1_rng = resid2_rng = None
        if train and self.dropout > 0.0:
            if rng is None:
                raise ValueError("TransformerBlock1D requires rng when dropout > 0 and train=True")
            attn_rng, ffn_rng, resid1_rng, resid2_rng = jax.random.split(rng, 4)

        h = self.norm1(x)
        h = self.attn(h, deterministic=deterministic, dropout_rng=attn_rng)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=resid1_rng)
        x = x + h

        h = self.norm2(x)
        h = self.ffn_in(h)
        h = nn.gelu(h, approximate=True)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=ffn_rng)
        h = self.ffn_out(h)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=resid2_rng)
        return x + h
