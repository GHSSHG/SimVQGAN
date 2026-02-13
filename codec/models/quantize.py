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
    diveq_sigma2: float = 1e-3
    search_chunk_size: int = 2048
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

    def _chunked_search(self, z_flat: jnp.ndarray, codebook: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
        """Exact nearest neighbor search with bounded peak memory.

        Instead of materializing (N, K) distances, this scans codebook chunks
        and keeps only current best distance/index for each latent.
        """
        K = int(self.codebook_size)
        C = max(1, int(chunk_size))
        if C >= K:
            return self._full_search(z_flat, codebook)

        n_blocks = (K + C - 1) // C
        pad = n_blocks * C - K
        if pad > 0:
            codebook = jnp.pad(codebook, ((0, pad), (0, 0)))
        blocks = codebook.reshape(n_blocks, C, self.code_dim)
        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)

        z_search = z_flat.astype(codebook.dtype)
        z_norm = jnp.sum(z_search**2, axis=1, keepdims=True)
        arange_c = jnp.arange(C, dtype=jnp.int32)
        inf = jnp.asarray(jnp.inf, dtype=z_norm.dtype)

        def _scan_step(carry, xs):
            best_dist, best_idx = carry
            block, block_id = xs
            offset = block_id * C
            emb_norm = jnp.sum(block**2, axis=1)
            dots = jnp.dot(z_search, block.T)
            dists = z_norm + emb_norm[None, :] - 2.0 * dots

            cand_idx_full = offset + arange_c
            valid = cand_idx_full < K
            dists = jnp.where(valid[None, :], dists, inf)

            chunk_arg = jnp.argmin(dists, axis=1).astype(jnp.int32)
            chunk_min = jnp.take_along_axis(dists, chunk_arg[:, None], axis=1)[:, 0]
            cand_idx = offset + chunk_arg

            better = chunk_min < best_dist
            best_dist = jnp.where(better, chunk_min, best_dist)
            best_idx = jnp.where(better, cand_idx, best_idx)
            return (best_dist, best_idx), None

        init_best_dist = jnp.full((z_flat.shape[0],), jnp.inf, dtype=z_norm.dtype)
        init_best_idx = jnp.zeros((z_flat.shape[0],), dtype=jnp.int32)
        (_, best_idx), _ = jax.lax.scan(_scan_step, (init_best_dist, init_best_idx), (blocks, block_ids))
        return best_idx

    def _project_codebook(self) -> jnp.ndarray:
        """Apply shared projection to codebook (stored in input space)."""
        emb = jax.lax.stop_gradient(self.codebook_var.value).astype(self.param_dtype)
        return jnp.dot(emb, self.W) + self.proj_bias

    def __call__(
        self,
        z: jnp.ndarray,
        *,
        rng: jax.random.KeyArray,
        train: bool = False,
        collect_codebook_stats: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        z_flat, (B, T) = self._prepare_latents(z)
        codebook = self._project_codebook()  # (K, D)
        indices = self._chunked_search(z_flat, codebook, self.search_chunk_size)
        quant_flat = jnp.take(codebook, indices, axis=0)

        quant = quant_flat.reshape(B, T, self.code_dim)
        z_reshaped = z_flat.reshape(B, T, self.code_dim)
        diff = quant - z_reshaped
        eps = jnp.asarray(1e-8, dtype=z_reshaped.dtype)
        diff_norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        diff_norm_safe = jnp.maximum(diff_norm, eps)
        sigma2 = max(0.0, float(self.diveq_sigma2))

        if train:
            if sigma2 > 0.0:
                noise_std = jnp.asarray(sigma2**0.5, dtype=z_reshaped.dtype)
                noise = noise_std * jax.random.normal(rng, shape=diff.shape, dtype=z_reshaped.dtype)
                diff_dir = noise + diff
            else:
                diff_dir = diff
            diff_dir_norm = jnp.linalg.norm(diff_dir, axis=-1, keepdims=True)
            diff_dir_norm = jnp.maximum(diff_dir_norm, eps)
            direction = jax.lax.stop_gradient(diff_dir / diff_dir_norm)
            z_q = z_reshaped + diff_norm_safe * direction
        else:
            # Hard nearest-codeword quantization for inference.
            z_q = quant

        # DiVeQ removes VQ auxiliary losses (commit/codebook); training relies on main task loss.
        commit_loss = jnp.asarray(0.0, dtype=self.dtype)

        indices_bt = indices.reshape(B, T)
        counts = None
        total_tokens = None
        avg_probs = None
        if collect_codebook_stats:
            flat_indices = indices_bt.reshape(-1).astype(jnp.int32)
            counts = _bincount(flat_indices, length=self.codebook_size, dtype=self.dtype)
            total_tokens = jnp.maximum(jnp.sum(counts), jnp.asarray(1.0, dtype=self.dtype))
            avg_probs = counts / total_tokens
            safe_probs = jnp.where(avg_probs > 0, avg_probs, jnp.asarray(1.0, dtype=self.dtype))
            perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(safe_probs + 1e-10)))
            usage_ratio = jnp.mean((counts > 0).astype(self.dtype))
        else:
            perplexity = jnp.asarray(0.0, dtype=self.dtype)
            usage_ratio = jnp.asarray(0.0, dtype=self.dtype)

        info = {
            "commit_loss": commit_loss.astype(self.dtype),
            "perplexity": perplexity.astype(self.dtype),
            "avg_probs": (avg_probs.astype(self.dtype) if avg_probs is not None else None),
            "usage_ratio": usage_ratio,
            "token_counts": (counts.astype(self.dtype) if counts is not None else None),
            "total_tokens": (total_tokens.astype(self.dtype) if total_tokens is not None else None),
            "indices": indices_bt,
        }
        return z_q, info
