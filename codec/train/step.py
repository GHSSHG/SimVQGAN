from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .losses import compute_generator_losses, hinge_d_loss, l1_time_loss


@partial(jax.jit, static_argnames=("collect_codebook_stats",))
def compute_grads(
    gen_state,
    disc_state,
    batch,
    rng,
    loss_weights,
    disc_mask: float,
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
            total_loss, logs = compute_generator_losses(
                y=batch,
                y_hat=wave_hat,
                fake_map=fake_map,
                real_feats=real_feats,
                fake_feats=fake_feats,
                commit_loss=outs["enc"]["commit_loss"],
                weights=weights,
            )
            loss_dtype = batch.dtype
            total_loss = jnp.asarray(total_loss, dtype=loss_dtype)
            logs = {k: jnp.asarray(v, dtype=loss_dtype) for k, v in logs.items()}
            return total_loss, logs

        def _gen_loss_without_disc(_):
            l_time = l1_time_loss(batch, wave_hat)
            dtype = l_time.dtype
            commit_loss = jnp.asarray(outs["enc"]["commit_loss"], dtype=dtype)
            w_recon = jnp.asarray(loss_weights.get("time_l1", 1.0), dtype=dtype)
            w_commit = jnp.asarray(loss_weights.get("commit", 1.0), dtype=dtype)
            zero = jnp.asarray(0.0, dtype=dtype)
            recon_term = w_recon * l_time
            commit_term = w_commit * commit_loss
            total = recon_term + commit_term
            logs = {
                "total_loss": total,
                "reconstruct_loss": l_time,
                "commit_loss": commit_loss,
                "weighted_reconstruct_loss": recon_term,
                "weighted_commit_loss": commit_term,
                "gan_raw_loss": zero,
                "weighted_gan_loss": zero,
                "feature_loss": zero,
                "weighted_feature_loss": zero,
            }
            return total, logs

        total_loss, logs = jax.lax.cond(
            disc_mask > 0.0,
            _gen_loss_with_disc,
            _gen_loss_without_disc,
            operand=None,
        )

        logs = dict(logs)
        if collect_codebook_stats:
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
