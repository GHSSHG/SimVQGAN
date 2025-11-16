from __future__ import annotations

import jax
import jax.numpy as jnp

from .losses import compute_generator_losses, hinge_d_loss


@jax.jit
def compute_grads(
    gen_state,
    disc_state,
    batch,
    rng,
    loss_weights,
    disc_factor: float,
):
    def gen_loss_fn(params):
        vq_in = gen_state.vq_vars if gen_state.vq_vars is not None else {}
        outs, mutable = gen_state.apply_fn(
            {"params": params, "vq": vq_in},
            batch,
            train=True,
            offset=0,
            rng=rng,
            mutable=["vq"],
        )
        vq_vars = mutable.get("vq", None)
        wave_hat = outs["wave_hat"]
        fake_map, fake_feats = disc_state.apply_fn(
            {"params": disc_state.params},
            wave_hat,
            train=True,
            return_features=True,
        )
        real_map, real_feats = disc_state.apply_fn(
            {"params": disc_state.params},
            batch,
            train=True,
            return_features=True,
        )
        weights = dict(loss_weights)
        gan_scale = jnp.asarray(weights.get("gan", 0.1), dtype=jnp.float32)
        disc_scale = jnp.asarray(disc_factor, dtype=jnp.float32)
        weights["gan"] = gan_scale * disc_scale
        total_loss, logs = compute_generator_losses(
            y=batch,
            y_hat=wave_hat,
            fake_map=fake_map,
            real_feats=real_feats,
            fake_feats=fake_feats,
            commit_loss=outs["enc"]["commit_loss"],
            weights=weights,
        )
        logs = dict(logs)
        logs["perplexity"] = outs["enc"].get("perplexity", jnp.array(0.0))
        usage_ratio = outs["enc"].get("usage_ratio")
        if usage_ratio is not None:
            logs["code_usage"] = usage_ratio
        token_counts = outs["enc"].get("token_counts")
        if token_counts is not None:
            logs["_code_hist_counts"] = token_counts
        total_tokens = outs["enc"].get("total_tokens")
        if total_tokens is not None:
            logs["_code_hist_total"] = total_tokens
        aux = {"logs": logs, "wave_hat": wave_hat, "vq_vars": vq_vars}
        return total_loss, aux

    (g_loss, aux), g_grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(gen_state.params)

    def disc_loss_fn(disc_params):
        real_map = disc_state.apply_fn({"params": disc_params}, batch, train=True)
        fake_map = disc_state.apply_fn(
            {"params": disc_params},
            jax.lax.stop_gradient(aux["wave_hat"]),
            train=True,
        )
        loss = hinge_d_loss(real_map, fake_map)
        logs = {"disc_loss": loss}
        return loss, logs

    (d_loss, d_logs), d_grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(disc_state.params)
    logs = dict(aux["logs"])
    logs.update(d_logs)
    logs["disc_loss"] = d_loss
    return g_grads, d_grads, logs, aux["vq_vars"]
