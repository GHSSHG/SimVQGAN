from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


_BINCOUNT_SUPPORTS_DTYPE: bool | None = None


def _bincount(data: jnp.ndarray, *, length: int, dtype: Any) -> jnp.ndarray:
    global _BINCOUNT_SUPPORTS_DTYPE
    if _BINCOUNT_SUPPORTS_DTYPE is None:
        try:
            jnp.bincount(jnp.array([0], dtype=jnp.int32), length=1, dtype=dtype)
            _BINCOUNT_SUPPORTS_DTYPE = True
        except TypeError:
            _BINCOUNT_SUPPORTS_DTYPE = False
    if _BINCOUNT_SUPPORTS_DTYPE:
        return jnp.bincount(data, length=length, dtype=dtype)
    counts = jnp.bincount(data, length=length)
    if counts.dtype != dtype:
        counts = counts.astype(dtype)
    return counts


class SimVQ1D(nn.Module):
    """SimVQ quantizer adapted for 1D latents (B, T, C)."""

    codebook_size: int
    code_dim: int
    beta: float = 0.25
    legacy_beta: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        embed_init = nn.initializers.normal(stddev=self.code_dim**-0.5)
        # SimVQ keeps the base embeddings fixed and only trains W to adapt latents.
        self.codebook_var = self.variable(
            "vq",
            "codebook",
            lambda: embed_init(self.make_rng("params"), (self.codebook_size, self.code_dim)).astype(self.dtype),
        )

        proj_init = nn.initializers.orthogonal()
        self.W = self.param(
            "W",
            proj_init,
            (self.code_dim, self.code_dim),
            self.param_dtype,
        )
        self.proj_bias = self.param(
            "proj_bias",
            nn.initializers.zeros,
            (self.code_dim,),
            self.param_dtype,
        )

    def _prepare_latents(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        if z.ndim != 3:
            raise ValueError(f"SimVQ1D expects (B,T,C) latents, got {z.shape}")
        B, T, C = z.shape
        if C != self.code_dim:
            raise ValueError(f"Latent dim {C} != code_dim {self.code_dim}")
        return z.reshape(B * T, C), (B, T)

    def _full_search(self, z_flat: jnp.ndarray, codebook: jnp.ndarray) -> jnp.ndarray:
        """Exact nearest neighbor search over the full codebook."""
        z_norm = jnp.sum(z_flat**2, axis=1, keepdims=True)
        emb_norm = jnp.sum(codebook**2, axis=1)
        dots = jnp.dot(z_flat, codebook.T)
        dists = z_norm + emb_norm[None, :] - 2.0 * dots
        return jnp.argmin(dists, axis=1)

    def _project_codebook(self) -> jnp.ndarray:
        """Apply shared projection to codebook (stored in input space)."""
        emb = jax.lax.stop_gradient(self.codebook_var.value).astype(self.param_dtype)
        return jnp.dot(emb, self.W) + self.proj_bias

    def __call__(self, z: jnp.ndarray, *, rng: jax.random.KeyArray, train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        del rng  # deterministic quantization
        z_flat, (B, T) = self._prepare_latents(z)
        codebook = self._project_codebook()  # (K, D)
        indices = self._full_search(z_flat, codebook)
        quant_flat = jnp.take(codebook, indices, axis=0)

        quant = quant_flat.reshape(B, T, self.code_dim)
        z_reshaped = z_flat.reshape(B, T, self.code_dim)
        z_stop = jax.lax.stop_gradient(z_reshaped)
        q_stop = jax.lax.stop_gradient(quant)

        if self.legacy_beta:
            commit = jnp.mean((q_stop - z_reshaped) ** 2)
            codebook_loss = self.beta * jnp.mean((quant - z_stop) ** 2)
        else:
            commit = self.beta * jnp.mean((q_stop - z_reshaped) ** 2)
            codebook_loss = jnp.mean((quant - z_stop) ** 2)
        commit_loss = commit + codebook_loss

        z_q = z_reshaped + jax.lax.stop_gradient(quant - z_reshaped)

        indices_bt = indices.reshape(B, T)
        flat_indices = indices_bt.reshape(-1).astype(jnp.int32)
        counts = _bincount(flat_indices, length=self.codebook_size, dtype=self.dtype)
        total_tokens = jnp.maximum(jnp.sum(counts), jnp.asarray(1.0, dtype=self.dtype))
        avg_probs = counts / total_tokens
        safe_probs = jnp.where(avg_probs > 0, avg_probs, jnp.asarray(1.0, dtype=self.dtype))
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(safe_probs + 1e-10)))
        usage_ratio = jnp.mean((counts > 0).astype(self.dtype))

        info = {
            "commit_loss": commit_loss.astype(self.dtype),
            "perplexity": perplexity.astype(self.dtype),
            "avg_probs": avg_probs.astype(self.dtype),
            "usage_ratio": usage_ratio,
            "token_counts": counts.astype(self.dtype),
            "total_tokens": total_tokens.astype(self.dtype),
            "indices": indices_bt,
        }
        return z_q, info
