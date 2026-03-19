from __future__ import annotations

import functools
from typing import Any, Callable

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


def _apply_rope(
    x: jnp.ndarray,
    *,
    base: float,
    positions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    head_dim = int(x.shape[-1])
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
    rope_dtype = jnp.float32
    if positions is None:
        seq_len = int(x.shape[-3])
        positions = jnp.arange(seq_len, dtype=rope_dtype)
    else:
        positions = jnp.asarray(positions, dtype=rope_dtype)
    inv_freq = 1.0 / (float(base) ** (jnp.arange(0, head_dim, 2, dtype=rope_dtype) / float(head_dim)))
    angles = positions[..., None] * inv_freq
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)
    expand_prefix = max(0, x.ndim - sin.ndim - 1)
    sin = sin.reshape((1,) * expand_prefix + sin.shape[:-1] + (1, sin.shape[-1]))
    cos = cos.reshape((1,) * expand_prefix + cos.shape[:-1] + (1, cos.shape[-1]))
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


def _xavier_normal_init(gain: float = 1.0) -> Callable[[jax.Array, tuple[int, ...], Any], jnp.ndarray]:
    def _init(key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jnp.ndarray:
        if len(shape) < 2:
            raise ValueError(f"Xavier initialization expects rank >= 2, got shape={shape}")
        fan_in = float(shape[-2])
        fan_out = float(shape[-1])
        std = float(gain) * jnp.sqrt(2.0 / (fan_in + fan_out))
        return jax.random.normal(key, shape, dtype=dtype) * std.astype(dtype)

    return _init


def _dorado_qkv_kernel_init(deepnorm_beta: float) -> Callable[[jax.Array, tuple[int, ...], Any], jnp.ndarray]:
    qk_init = _xavier_normal_init(1.0)
    v_init = _xavier_normal_init(deepnorm_beta)

    def _init(key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jnp.ndarray:
        if len(shape) != 2:
            raise ValueError(f"Dorado qkv init expects rank-2 Dense kernel, got shape={shape}")
        in_dim, out_dim = int(shape[0]), int(shape[1])
        if out_dim % 3 != 0:
            raise ValueError(f"Dorado qkv init expects output dim divisible by 3, got {out_dim}")
        qkv_dim = out_dim // 3
        key_qk, key_v = jax.random.split(key)
        qk_kernel = qk_init(key_qk, (in_dim, 2 * qkv_dim), dtype)
        v_kernel = v_init(key_v, (in_dim, qkv_dim), dtype)
        return jnp.concatenate((qk_kernel, v_kernel), axis=-1)

    return _init


def _resolve_local_attn_window(window_size: int) -> tuple[int, int]:
    total = max(1, int(window_size))
    left = max(0, (total // 2) - 1)
    right = max(0, total - left - 1)
    return left, right


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


class RMSNorm(nn.Module):
    dim: int
    eps: float | None = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expects trailing dim {self.dim}, got {x.shape[-1]}")
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        x_fp32 = x.astype(jnp.float32)
        if self.eps is None:
            eps = jnp.finfo(x_fp32.dtype).eps
        else:
            eps = float(self.eps)
        rms = jnp.sqrt(jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True) + eps)
        y = x_fp32 / rms
        y = y.astype(self.dtype)
        return y * scale.astype(self.dtype)


class LocalTransformerBlock1D(nn.Module):
    """Dorado-style local-attention Transformer block for (B, T, C) latents."""

    dim: int
    num_heads: int = 4
    window_size: int = 256
    query_chunk_size: int = 128
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ffn_activation: str = "swiglu"
    attention_backend: str = "jax_cudnn"
    use_rope: bool = True
    rope_base: float = 10000.0
    deepnorm_alpha: float = 2.4494897
    deepnorm_beta: float = 0.2886751
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        if int(self.window_size) <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if int(self.query_chunk_size) <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {self.query_chunk_size}")
        hidden_dim = int(round(self.dim * float(self.mlp_ratio)))
        if hidden_dim <= 0:
            raise ValueError(f"mlp hidden dim must be positive, got {hidden_dim}")
        if self.use_rope and (self.dim // self.num_heads) % 2 != 0:
            raise ValueError(
                f"RoPE requires an even per-head dim, got dim={self.dim}, num_heads={self.num_heads}"
            )
        activation = str(self.ffn_activation).strip().lower()
        if activation != "swiglu":
            raise ValueError(
                f"LocalTransformerBlock1D matches Dorado and only supports ffn_activation='swiglu', got {self.ffn_activation!r}."
            )
        del activation
        self._left_window, self._right_window = _resolve_local_attn_window(self.window_size)
        self._head_dim = self.dim // self.num_heads
        self.norm1 = RMSNorm(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
        self.qkv_proj = nn.Dense(
            3 * self.dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_dorado_qkv_kernel_init(float(self.deepnorm_beta)),
            name="Wqkv",
        )
        self.out_proj = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_xavier_normal_init(float(self.deepnorm_beta)),
            name="out_proj",
        )
        self.norm2 = RMSNorm(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm2")
        self.ffn_in = nn.Dense(
            hidden_dim * 2,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_xavier_normal_init(float(self.deepnorm_beta)),
            name="ff_fc1",
        )
        self.ffn_out = nn.Dense(
            self.dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_xavier_normal_init(float(self.deepnorm_beta)),
            name="ff_fc2",
        )

    def _local_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        batch, seq_len, _, head_dim = q.shape
        query_chunk = min(max(1, int(self.query_chunk_size)), seq_len)
        query_pad = (-seq_len) % query_chunk
        q_padded = jnp.pad(q, ((0, 0), (0, query_pad), (0, 0), (0, 0)))
        out = jnp.zeros_like(q_padded, dtype=jnp.float32)

        left_window = int(self._left_window)
        right_window = int(self._right_window)
        key_span = query_chunk + left_window + right_window
        k_padded = jnp.pad(k, ((0, 0), (left_window, right_window + query_pad), (0, 0), (0, 0)))
        v_padded = jnp.pad(v, ((0, 0), (left_window, right_window + query_pad), (0, 0), (0, 0)))
        query_valid = jnp.arange(seq_len + query_pad, dtype=jnp.int32) < seq_len
        key_positions = jnp.arange(-left_window, seq_len + right_window + query_pad, dtype=jnp.int32)
        key_valid = (key_positions >= 0) & (key_positions < seq_len)
        scale = jnp.asarray(head_dim**-0.5, dtype=jnp.float32)
        mask_fill = jnp.asarray(-1e30, dtype=jnp.float32)

        for q_start in range(0, seq_len + query_pad, query_chunk):
            q_block = jax.lax.dynamic_slice(q_padded, (0, q_start, 0, 0), (batch, query_chunk, self.num_heads, head_dim))
            k_block = jax.lax.dynamic_slice(k_padded, (0, q_start, 0, 0), (batch, key_span, self.num_heads, head_dim))
            v_block = jax.lax.dynamic_slice(v_padded, (0, q_start, 0, 0), (batch, key_span, self.num_heads, head_dim))
            q_positions = q_start + jnp.arange(query_chunk, dtype=jnp.int32)
            k_positions = jax.lax.dynamic_slice(key_positions, (q_start,), (key_span,))
            q_mask = jax.lax.dynamic_slice(query_valid, (q_start,), (query_chunk,))
            k_mask = jax.lax.dynamic_slice(key_valid, (q_start,), (key_span,))

            scores = jnp.einsum(
                "bqhd,bkhd->bhqk",
                q_block.astype(jnp.float32),
                k_block.astype(jnp.float32),
            )
            scores = scores * scale
            block_mask = (
                q_mask[None, None, :, None]
                & k_mask[None, None, None, :]
                & (k_positions[None, None, None, :] >= (q_positions[None, None, :, None] - left_window))
                & (k_positions[None, None, None, :] <= (q_positions[None, None, :, None] + right_window))
            )
            scores = jnp.where(block_mask, scores, mask_fill)
            attn = nn.softmax(scores, axis=-1)
            block_out = jnp.einsum("bhqk,bkhd->bqhd", attn, v_block.astype(jnp.float32))
            block_out = block_out * q_mask[None, :, None, None].astype(block_out.dtype)
            out = jax.lax.dynamic_update_slice(out, block_out, (0, q_start, 0, 0))

        return out[:, :seq_len, :, :].astype(self.dtype)

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"LocalTransformerBlock1D expects (B,T,C), got {x.shape}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dim {x.shape[-1]} != block dim {self.dim}")

        ffn_rng = resid1_rng = resid2_rng = None
        if train and self.dropout > 0.0:
            if rng is None:
                raise ValueError("LocalTransformerBlock1D requires rng when dropout > 0 and train=True")
            ffn_rng, resid1_rng, resid2_rng = jax.random.split(rng, 3)

        qkv = self.qkv_proj(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self._head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)
        if self.use_rope:
            q = _apply_rope(q, base=float(self.rope_base))
            k = _apply_rope(k, base=float(self.rope_base))
        h = self._local_attention(q, k, v).reshape(x.shape[0], x.shape[1], self.dim)
        h = self.out_proj(h)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=resid1_rng)
        alpha = jnp.asarray(float(self.deepnorm_alpha), dtype=h.dtype)
        x = self.norm1(h + (x.astype(h.dtype) * alpha))

        h = self.ffn_in(x)
        h_value, h_gate = jnp.split(h, 2, axis=-1)
        h = h_value * nn.silu(h_gate)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=ffn_rng)
        h = self.ffn_out(h)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=resid2_rng)
        x = self.norm2(h + (x.astype(h.dtype) * alpha))
        return x.astype(self.dtype)


class SwinTransformerBlock1D(nn.Module):
    """Windowed 1D Swin-style block for (B, T, C) latents."""

    dim: int
    num_heads: int = 4
    window_size: int = 256
    shift_size: int = 0
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
        if int(self.window_size) <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if int(self.shift_size) < 0:
            raise ValueError(f"shift_size must be non-negative, got {self.shift_size}")
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
        backend = str(self.attention_backend).strip().lower()
        valid_backends = {"flax", "jax_cudnn"}
        if backend not in valid_backends:
            raise ValueError(
                f"Unsupported attention_backend={self.attention_backend!r}; choose from {sorted(valid_backends)}."
            )
        self._attention_backend = backend
        self._ffn_activation = activation
        self.norm1 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
        self.q_proj = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q_proj",
        )
        self.k_proj = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="k_proj",
        )
        self.v_proj = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="v_proj",
        )
        self.out_proj = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out_proj",
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

    def _window_attention(
        self,
        x: jnp.ndarray,
        *,
        train: bool,
        rng: jax.Array | None,
    ) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        window_size = min(max(1, int(self.window_size)), max(1, seq_len))
        shift_size = int(self.shift_size)
        if window_size >= seq_len:
            shift_size = 0
        else:
            shift_size = min(max(0, shift_size), window_size - 1)

        positions = jnp.arange(seq_len, dtype=jnp.int32)
        group_ids = positions // window_size
        valid = jnp.ones((seq_len,), dtype=jnp.bool_)
        h = x
        if shift_size > 0:
            h = jnp.roll(h, shift=-shift_size, axis=1)
            positions = jnp.roll(positions, shift=-shift_size)
            group_ids = jnp.roll(group_ids, shift=-shift_size)

        pad_len = (-seq_len) % window_size
        if pad_len > 0:
            h = jnp.pad(h, ((0, 0), (0, pad_len), (0, 0)))
            positions = jnp.pad(positions, (0, pad_len), constant_values=0)
            group_ids = jnp.pad(group_ids, (0, pad_len), constant_values=-1)
            valid = jnp.pad(valid, (0, pad_len), constant_values=False)
        padded_len = seq_len + pad_len
        num_windows = padded_len // window_size

        h_windows = h.reshape(batch, num_windows, window_size, self.dim)
        h_windows = h_windows.reshape(batch * num_windows, window_size, self.dim)
        pos_windows = jnp.broadcast_to(positions[None, :], (batch, padded_len))
        pos_windows = pos_windows.reshape(batch, num_windows, window_size)
        pos_windows = pos_windows.reshape(batch * num_windows, window_size)
        group_windows = jnp.broadcast_to(group_ids[None, :], (batch, padded_len))
        group_windows = group_windows.reshape(batch, num_windows, window_size)
        group_windows = group_windows.reshape(batch * num_windows, window_size)
        valid_windows = jnp.broadcast_to(valid[None, :], (batch, padded_len))
        valid_windows = valid_windows.reshape(batch, num_windows, window_size)
        valid_windows = valid_windows.reshape(batch * num_windows, window_size)

        head_dim = self.dim // self.num_heads
        q = self.q_proj(h_windows).reshape(batch * num_windows, window_size, self.num_heads, head_dim)
        k = self.k_proj(h_windows).reshape(batch * num_windows, window_size, self.num_heads, head_dim)
        v = self.v_proj(h_windows).reshape(batch * num_windows, window_size, self.num_heads, head_dim)
        if self.use_rope:
            q = _apply_rope(q, base=float(self.rope_base), positions=pos_windows)
            k = _apply_rope(k, base=float(self.rope_base), positions=pos_windows)

        attn_mask = None
        if shift_size > 0 or pad_len > 0:
            same_group = group_windows[:, :, None] == group_windows[:, None, :]
            valid_pairs = valid_windows[:, :, None] & valid_windows[:, None, :]
            attn_mask = (same_group & valid_pairs)[:, None, :, :]

        deterministic = (not train) or (self.dropout <= 0.0)
        implementation = "cudnn" if self._attention_backend == "jax_cudnn" and attn_mask is None else None
        attn = _jax_attention_fn(
            q,
            k,
            v,
            mask=attn_mask,
            dropout_rng=None,
            dropout_rate=0.0,
            deterministic=deterministic,
            dtype=self.dtype,
            implementation=implementation,
        )
        attn = attn.reshape(batch * num_windows, window_size, self.dim)
        attn = self.out_proj(attn)
        attn = attn.reshape(batch, num_windows, window_size, self.dim)
        attn = attn.reshape(batch, padded_len, self.dim)
        if shift_size > 0:
            attn = jnp.roll(attn, shift=shift_size, axis=1)
        if pad_len > 0:
            attn = attn[:, :seq_len, :]
        return attn

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"SwinTransformerBlock1D expects (B,T,C), got {x.shape}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dim {x.shape[-1]} != block dim {self.dim}")

        attn_rng = ffn_rng = resid1_rng = resid2_rng = None
        if train and self.dropout > 0.0:
            if rng is None:
                raise ValueError("SwinTransformerBlock1D requires rng when dropout > 0 and train=True")
            attn_rng, ffn_rng, resid1_rng, resid2_rng = jax.random.split(rng, 4)

        h = self.norm1(x)
        h = self._window_attention(h, train=train, rng=attn_rng)
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
