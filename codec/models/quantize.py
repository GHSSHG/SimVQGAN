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
    """SimVQ quantizer adapted for 1D latents (B, T, C).

    Enhancements:
      1) EMA-based codebook updates (no optimizer step needed).
      2) Chunked top-k coarse search to reduce nearest-neighbor cost on large codebooks.
    """

    codebook_size: int
    code_dim: int
    beta: float = 0.25
    legacy_beta: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    chunk_size: int = 512
    topk_chunks: int = 4
    search_batch_size: int = 2048

    def setup(self) -> None:
        embed_init = nn.initializers.normal(stddev=self.code_dim**-0.5)
        if self.use_ema:
            # Codebook + EMA stats live in mutable "vq" collection so the optimizer never touches them.
            self.codebook_var = self.variable(
                "vq",
                "codebook",
                lambda: embed_init(self.make_rng("params"), (self.codebook_size, self.code_dim)),
            )
            self.ema_cluster = self.variable(
                "vq",
                "ema_cluster_size",
                lambda: jnp.zeros((self.codebook_size,), dtype=self.dtype),
            )
            self.ema_embed = self.variable(
                "vq",
                "ema_embed",
                lambda: self.codebook_var.value.astype(self.dtype),
            )
        else:
            # Standard learnable codebook updated by optimizer when EMA disabled.
            self.codebook_param = self.param(
                "codebook",
                embed_init,
                (self.codebook_size, self.code_dim),
                self.param_dtype,
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

    def _chunk_topk_search(self, z_flat: jnp.ndarray, codebook: jnp.ndarray) -> jnp.ndarray:
        """Two-stage search: pick nearest chunks via top-k, then brute force within them.

        Uses batching over tokens to cap peak memory during per-token chunk gathers.
        """
        K, D = codebook.shape
        if self.chunk_size <= 0 or self.topk_chunks <= 0 or self.chunk_size >= K:
            # Fallback to exact search
            z_norm = jnp.sum(z_flat**2, axis=1, keepdims=True)
            emb_norm = jnp.sum(codebook**2, axis=1)
            dots = jnp.dot(z_flat, codebook.T)
            dists = z_norm + emb_norm[None, :] - 2.0 * dots
            return jnp.argmin(dists, axis=1)

        chunk = min(self.chunk_size, K)
        num_chunks = (K + chunk - 1) // chunk
        pad = num_chunks * chunk - K
        if pad > 0:
            pad_vals = jnp.zeros((pad, D), dtype=codebook.dtype)
            codebook_pad = jnp.concatenate([codebook, pad_vals], axis=0)
        else:
            codebook_pad = codebook
        chunked = codebook_pad.reshape(num_chunks, chunk, D)
        chunk_sizes = jnp.full((num_chunks,), chunk, dtype=jnp.int32)
        if pad > 0:
            chunk_sizes = chunk_sizes.at[-1].set(chunk - pad)

        # coarse distances to chunk means (mask padding when computing means)
        chunk_sums = jnp.sum(chunked, axis=1)
        chunk_counts = chunk_sizes.astype(self.dtype)[:, None]
        chunk_means = chunk_sums / jnp.maximum(chunk_counts, jnp.asarray(1.0, dtype=self.dtype))

        z_exp = z_flat[:, None, :]  # (N, 1, D)
        d_chunk = jnp.sum((z_exp - chunk_means[None, :, :]) ** 2, axis=2)  # (N, num_chunks)
        topk = min(self.topk_chunks, num_chunks)
        _, topk_idx = jax.lax.top_k(-d_chunk, k=topk)  # (N, topk)

        chunk_sizes_f = chunk_sizes
        arange_chunk = jnp.arange(chunk)

        def _search_one(z_i, chunk_ids):
            selected = chunked[chunk_ids, :, :]  # (topk, chunk, D)
            flat_selected = selected.reshape(-1, D)
            valid_mask = (arange_chunk[None, :] < chunk_sizes_f[chunk_ids][:, None]).reshape(-1)
            dists = jnp.sum((flat_selected - z_i[None, :]) ** 2, axis=1)
            dists = jnp.where(valid_mask, dists, jnp.inf)
            local_idx = jnp.argmin(dists)
            chunk_offset = local_idx // chunk
            elem_offset = local_idx % chunk
            global_idx = chunk_ids[chunk_offset] * chunk + elem_offset
            return jnp.minimum(global_idx, K - 1)

        def search_batch(z_batch: jnp.ndarray) -> jnp.ndarray:
            z_exp = z_batch[:, None, :]  # (Nb,1,D)
            d_chunk = jnp.sum((z_exp - chunk_means[None, :, :]) ** 2, axis=2)  # (Nb, num_chunks)
            topk = min(self.topk_chunks, num_chunks)
            _, tk_idx = jax.lax.top_k(-d_chunk, k=topk)  # (Nb, topk)
            return jax.vmap(_search_one)(z_batch, tk_idx)

        N = z_flat.shape[0]
        batch = max(1, int(self.search_batch_size))
        num_batches = (N + batch - 1) // batch
        pad = num_batches * batch - N
        if pad > 0:
            z_flat_pad = jnp.pad(z_flat, ((0, pad), (0, 0)))
        else:
            z_flat_pad = z_flat

        # Sequentially process fixed-size chunks so peak memory stays bounded by `search_batch_size`.
        out = jnp.zeros((num_batches * batch,), dtype=jnp.int32)

        def body(out_arr, i):
            start = i * batch
            z_chunk = jax.lax.dynamic_slice(z_flat_pad, (start, 0), (batch, D))
            idx_batch = search_batch(z_chunk)
            out_arr = jax.lax.dynamic_update_slice(out_arr, idx_batch, (start,))
            return out_arr, None

        out, _ = jax.lax.scan(body, out, jnp.arange(num_batches))
        return out[:N]

    def _project_codebook(self) -> jnp.ndarray:
        """Apply shared projection to codebook (stored in input space)."""
        if self.use_ema:
            emb = jax.lax.stop_gradient(self.codebook_var.value)
        else:
            emb = self.codebook_param
        return jnp.dot(emb, self.W) + self.proj_bias

    def __call__(self, z: jnp.ndarray, *, rng: jax.random.KeyArray, train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        del rng  # deterministic quantization
        z_flat, (B, T) = self._prepare_latents(z)
        codebook = self._project_codebook()  # (K, D)
        indices = self._chunk_topk_search(z_flat, codebook)
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

        if self.use_ema and train:
            decay = jnp.asarray(self.ema_decay, dtype=self.dtype)
            one_minus = jnp.asarray(1.0 - self.ema_decay, dtype=self.dtype)
            W = jax.lax.stop_gradient(self.W).astype(self.dtype)
            bias = jax.lax.stop_gradient(self.proj_bias).astype(self.dtype)
            rhs = z_flat.astype(self.dtype) - bias
            solve_mat = jnp.transpose(W) + self.ema_eps * jnp.eye(self.code_dim, dtype=self.dtype)
            z_unproj = jnp.linalg.solve(solve_mat, rhs.T).T
            embed_sums = jnp.zeros((self.codebook_size, self.code_dim), dtype=self.dtype).at[flat_indices].add(z_unproj)
            ema_cluster_size = decay * self.ema_cluster.value + one_minus * counts
            ema_embed = decay * self.ema_embed.value + one_minus * embed_sums
            denom = ema_cluster_size + self.ema_eps
            new_codebook = ema_embed / denom[:, None]
            self.ema_cluster.value = ema_cluster_size
            self.ema_embed.value = ema_embed
            self.codebook_var.value = new_codebook.astype(self.param_dtype)

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
