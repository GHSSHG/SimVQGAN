from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .losses import compute_generator_losses, compute_reconstruction_losses, hinge_d_loss


@partial(
    jax.jit,
    static_argnames=("collect_codebook_stats", "stft_n_fft", "stft_hop_length", "stft_win_length"),
)
def compute_grads(
    gen_state,
    disc_state,
    batch,
    rng,
    loss_weights,
    disc_mask: float,
    *,
    stft_n_fft: int = 256,
    stft_hop_length: int = 64,
    stft_win_length: int = 256,
    collect_codebook_stats: bool = True,
):
    disc_mask = jnp.asarray(disc_mask, dtype=jnp.float32)

    def gen_loss_fn(params):
        vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
        outs = gen_state.apply_fn(
            {"params": params, "vq": vq_in},
            batch,
            train=True,
            offset=0,
            rng=rng,
            collect_codebook_stats=collect_codebook_stats,
        )
        vq_vars = gen_state.vq_vars
        wave_hat = outs["wave_hat"]
        log_q_z_dist = outs["enc"].get("log_q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))

        def _gen_loss_with_disc(_):
            fake_map, fake_feats = disc_state.apply_fn(
                {"params": disc_state.params},
                wave_hat,
                train=True,
            )
            _, real_feats = disc_state.apply_fn(
                {"params": disc_state.params},
                batch,
                train=True,
            )
            weights = dict(loss_weights)
            gan_w = jnp.asarray(weights.get("gan", 0.1), dtype=jnp.float32)
            feat_w = jnp.asarray(weights.get("feature", 0.0), dtype=jnp.float32)
            weights["gan"] = disc_mask * gan_w
            weights["feature"] = disc_mask * feat_w
            return compute_generator_losses(
                y=batch,
                y_hat=wave_hat,
                fake_map=fake_map,
                real_feats=real_feats,
                fake_feats=fake_feats,
                weights=weights,
                stft_n_fft=stft_n_fft,
                stft_hop_length=stft_hop_length,
                stft_win_length=stft_win_length,
            )

        def _gen_loss_without_disc(_):
            reconstruct, logs = compute_reconstruction_losses(
                y=batch,
                y_hat=wave_hat,
                weights=loss_weights,
                stft_n_fft=stft_n_fft,
                stft_hop_length=stft_hop_length,
                stft_win_length=stft_win_length,
            )
            zero = jnp.asarray(0.0, dtype=reconstruct.dtype)
            logs = dict(logs)
            logs.update(
                {
                    "total_loss": reconstruct,
                    "gan_loss_raw": zero,
                    "gan_loss": zero,
                    "feature_loss_raw": zero,
                    "feature_loss": zero,
                }
            )
            return reconstruct, logs

        total_loss, logs = jax.lax.cond(
            disc_mask > 0.0,
            _gen_loss_with_disc,
            _gen_loss_without_disc,
            operand=None,
        )

        logs = dict(logs)
        logs["q_z_dist"] = outs["enc"].get("q_z_dist", jnp.asarray(0.0, dtype=jnp.float32))
        logs["log_q_z_dist"] = log_q_z_dist
        if collect_codebook_stats:
            logs["perplexity"] = outs["enc"].get("perplexity", jnp.array(0.0))
            usage_ratio = outs["enc"].get("usage_ratio")
            if usage_ratio is not None:
                logs["code_usage"] = usage_ratio
        loss_dtype = batch.dtype
        total_loss = jnp.asarray(total_loss, dtype=loss_dtype)
        logs = {k: jnp.asarray(v, dtype=loss_dtype) for k, v in logs.items()}
        aux = {"logs": logs, "wave_hat": wave_hat, "vq_vars": vq_vars}
        return total_loss, aux

    (_g_loss, aux), g_grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(gen_state.params)

    def disc_loss_fn(disc_params):
        def _disc_loss_with_disc(_):
            real_map, _ = disc_state.apply_fn({"params": disc_params}, batch, train=True)
            fake_map, _ = disc_state.apply_fn(
                {"params": disc_params},
                jax.lax.stop_gradient(aux["wave_hat"]),
                train=True,
            )
            raw_loss = hinge_d_loss(real_map, fake_map)
            loss = disc_mask * raw_loss
            loss = jnp.asarray(loss, dtype=jnp.float32)
            raw_loss = jnp.asarray(raw_loss, dtype=jnp.float32)
            logs = {"disc_loss": loss, "disc_loss_raw": raw_loss, "disc_mask": disc_mask}
            return loss, logs

        def _disc_loss_without_disc(_):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            logs = {"disc_loss": zero, "disc_loss_raw": zero, "disc_mask": disc_mask}
            return zero, logs

        return jax.lax.cond(
            disc_mask > 0.0,
            _disc_loss_with_disc,
            _disc_loss_without_disc,
            operand=None,
        )

    (d_loss, d_logs), d_grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(disc_state.params)
    logs = dict(aux["logs"])
    logs.update(d_logs)
    logs["disc_loss"] = d_loss
    return g_grads, d_grads, logs, aux["vq_vars"]
