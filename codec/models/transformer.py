from __future__ import annotations

import functools
from typing import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.attention import dot_product_attention as _flax_dot_product_attention


def _dropout(x: jnp.ndarray, *, rate: float, train: bool, rng: jax.Array | None) -> jnp.ndarray:
    if (not train) or rate <= 0.0:
        return x
    if rng is None:
        raise ValueError("dropout requires rng when train=True")
    keep_prob = 1.0 - rate
    keep_mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return jnp.where(keep_mask, x / keep_prob, 0.0)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack((-x_odd, x_even), axis=-1)
    return rotated.reshape(x.shape)


def _apply_rope(x: jnp.ndarray, *, base: float) -> jnp.ndarray:
    head_dim = int(x.shape[-1])
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
    seq_len = int(x.shape[-3])
    rope_dtype = jnp.float32
    positions = jnp.arange(seq_len, dtype=rope_dtype)
    inv_freq = 1.0 / (float(base) ** (jnp.arange(0, head_dim, 2, dtype=rope_dtype) / float(head_dim)))
    angles = positions[:, None] * inv_freq[None, :]
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)[None, :, None, :]
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)[None, :, None, :]
    x_fp32 = x.astype(rope_dtype)
    x_rot = (x_fp32 * cos) + (_rotate_half(x_fp32) * sin)
    return x_rot.astype(x.dtype)


def _wrap_attention_fn_with_rope(
    attention_fn: Callable[..., jnp.ndarray],
    *,
    use_rope: bool,
    rope_base: float,
) -> Callable[..., jnp.ndarray]:
    if not use_rope:
        return attention_fn

    def _attention_with_rope(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        query = _apply_rope(query, base=rope_base)
        key = _apply_rope(key, base=rope_base)
        return attention_fn(query, key, value, **kwargs)

    return _attention_with_rope


def _jax_attention_fn(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jax.Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Any = None,
    precision: Any = None,
    module: nn.Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., jnp.ndarray] | None = None,
    qk_attn_weights_einsum: Callable[..., jnp.ndarray] | None = None,
    attn_weights_value_einsum: Callable[..., jnp.ndarray] | None = None,
    *,
    implementation: str | None = None,
) -> jnp.ndarray:
    del (
        broadcast_dropout,
        precision,
        module,
        force_fp32_for_softmax,
        einsum_dot_general,
        qk_attn_weights_einsum,
        attn_weights_value_einsum,
    )
    query_in = query
    key_in = key
    value_in = value
    query_seq_lengths = None
    key_value_seq_lengths = None
    query_len = int(query.shape[-3])
    if (
        implementation == "cudnn"
        and bias is None
        and mask is None
        and query.ndim == 4
        and key.ndim == 4
        and value.ndim == 4
    ):
        batch = int(query.shape[0])
        key_len = int(key.shape[1])
        padded_query_len = 1 << max(0, query_len - 1).bit_length()
        padded_key_len = 1 << max(0, key_len - 1).bit_length()
        if padded_query_len != query_len:
            query_in = jnp.pad(query, ((0, 0), (0, padded_query_len - query_len), (0, 0), (0, 0)))
        if padded_key_len != key_len:
            pad = padded_key_len - key_len
            key_in = jnp.pad(key, ((0, 0), (0, pad), (0, 0), (0, 0)))
            value_in = jnp.pad(value, ((0, 0), (0, pad), (0, 0), (0, 0)))
        query_seq_lengths = jnp.full((batch,), query_len, dtype=jnp.int32)
        key_value_seq_lengths = jnp.full((batch,), key_len, dtype=jnp.int32)
    out = jax.nn.dot_product_attention(
        query_in,
        key_in,
        value_in,
        bias=bias,
        mask=mask,
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=key_value_seq_lengths,
        implementation=implementation,
    )
    if query_in.shape[-3] != query_len:
        out = out[:, :query_len, :, :]
    if dtype is not None and out.dtype != dtype:
        out = out.astype(dtype)
    return _dropout(out, rate=float(dropout_rate), train=(not deterministic), rng=dropout_rng)


class TransformerBlock1D(nn.Module):
    """Pre-norm Transformer block for sequence latents shaped (B, T, C)."""

    dim: int
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ffn_activation: str = "gelu"
    attention_backend: str = "jax_cudnn"
    use_rope: bool = False
    rope_base: float = 10000.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        hidden_dim = int(round(self.dim * float(self.mlp_ratio)))
        if hidden_dim <= 0:
            raise ValueError(f"mlp hidden dim must be positive, got {hidden_dim}")
        if self.use_rope and (self.dim // self.num_heads) % 2 != 0:
            raise ValueError(
                f"RoPE requires an even per-head dim, got dim={self.dim}, num_heads={self.num_heads}"
            )
        activation = str(self.ffn_activation).strip().lower()
        if activation not in {"gelu", "swiglu"}:
            raise ValueError(f"Unsupported ffn_activation={self.ffn_activation!r}; use 'gelu' or 'swiglu'.")
        self._ffn_activation = activation
        backend = str(self.attention_backend).strip().lower()
        valid_backends = {"flax", "jax_cudnn"}
        if backend not in valid_backends:
            raise ValueError(
                f"Unsupported attention_backend={self.attention_backend!r}; choose from {sorted(valid_backends)}."
            )
        attention_fn: Callable[..., jnp.ndarray]
        if backend == "flax":
            attention_fn = _flax_dot_product_attention
        else:
            attention_fn = functools.partial(_jax_attention_fn, implementation="cudnn")
        attention_fn = _wrap_attention_fn_with_rope(
            attention_fn,
            use_rope=bool(self.use_rope),
            rope_base=float(self.rope_base),
        )

        self.norm1 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=float(self.dropout),
            attention_fn=attention_fn,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attn",
        )
        self.norm2 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm2")
        self.ffn_in = nn.Dense(
            hidden_dim * (2 if self._ffn_activation == "swiglu" else 1),
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
        if self._ffn_activation == "swiglu":
            h_value, h_gate = jnp.split(h, 2, axis=-1)
            h = h_value * nn.silu(h_gate)
        else:
            h = nn.gelu(h, approximate=True)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=ffn_rng)
        h = self.ffn_out(h)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=resid2_rng)
        return x + h
