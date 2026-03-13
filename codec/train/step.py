from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..dorado import DoradoPerceptualState
from .losses import compute_reconstruction_losses


def _extract_signal(batch):
    if isinstance(batch, dict):
        return batch["signal"]
    return batch


def _extract_dorado_metadata(batch):
    if not isinstance(batch, dict):
        return None, None
    if "pa_mean" not in batch or "pa_std" not in batch:
        return None, None
    return batch["pa_mean"], batch["pa_std"]


@partial(
    jax.jit,
    static_argnames=("collect_codebook_stats", "stft_loss_scales"),
)
def compute_grads(
    gen_state,
    batch,
    rng,
    loss_weights,
    *,
    stft_loss_scales: tuple[tuple[int, int, int], ...] = ((256, 256, 64),),
    dorado_perceptual_state: DoradoPerceptualState | None = None,
    collect_codebook_stats: bool = True,
):
    def gen_loss_fn(params):
        signal = _extract_signal(batch)
        pa_mean, pa_std = _extract_dorado_metadata(batch)
        vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
        outs = gen_state.apply_fn(
            {"params": params, "vq": vq_in},
            signal,
            train=True,
            offset=0,
            rng=rng,
            collect_codebook_stats=collect_codebook_stats,
        )
        vq_vars = gen_state.vq_vars
        wave_hat = outs["wave_hat"]
        total_loss, logs = compute_reconstruction_losses(
            y=signal,
            y_hat=wave_hat,
            weights=loss_weights,
            stft_loss_scales=stft_loss_scales,
            pa_mean=pa_mean,
            pa_std=pa_std,
            step=gen_state.step,
            dorado_perceptual_state=dorado_perceptual_state,
        )
        logs = dict(logs)
        logs["total_loss"] = total_loss
        logs["q_z_dist"] = outs["enc"].get("q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        logs["log_q_z_dist"] = outs["enc"].get("log_q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        if collect_codebook_stats:
            logs["perplexity"] = outs["enc"].get("perplexity", jnp.array(0.0))
            usage_ratio = outs["enc"].get("usage_ratio")
            if usage_ratio is not None:
                logs["code_usage"] = usage_ratio
        loss_dtype = signal.dtype
        total_loss = jnp.asarray(total_loss, dtype=loss_dtype)
        logs = {k: jnp.asarray(v, dtype=loss_dtype) for k, v in logs.items()}
        aux = {"logs": logs, "vq_vars": vq_vars}
        return total_loss, aux

    (_g_loss, aux), g_grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(gen_state.params)
    return g_grads, aux["logs"], aux["vq_vars"]
