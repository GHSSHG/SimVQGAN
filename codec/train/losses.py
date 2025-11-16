from __future__ import annotations

from typing import Dict, Sequence

import jax.numpy as jnp


def l1_time_loss(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y - y_hat))


def map_mean(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def hinge_g_loss(fake_map: jnp.ndarray) -> jnp.ndarray:
    return -map_mean(fake_map).mean()


def hinge_d_loss(real_map: jnp.ndarray, fake_map: jnp.ndarray) -> jnp.ndarray:
    real_term = jnp.mean(jnp.maximum(0.0, 1.0 - real_map))
    fake_term = jnp.mean(jnp.maximum(0.0, 1.0 + fake_map))
    return real_term + fake_term


def feature_matching_loss(real_feats: Sequence[jnp.ndarray], fake_feats: Sequence[jnp.ndarray]) -> jnp.ndarray:
    losses = [jnp.mean(jnp.abs(r - f)) for r, f in zip(real_feats, fake_feats)]
    if not losses:
        return jnp.array(0.0)
    return jnp.mean(jnp.stack(losses))


def compute_generator_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    fake_map: jnp.ndarray,
    real_feats: Sequence[jnp.ndarray],
    fake_feats: Sequence[jnp.ndarray],
    commit_loss: jnp.ndarray,
    weights: Dict[str, float],
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    l_time = l1_time_loss(y, y_hat)
    l_g = hinge_g_loss(fake_map)
    l_fm = feature_matching_loss(real_feats, fake_feats)
    dtype = l_time.dtype

    def _weight(name: str, default: float) -> jnp.ndarray:
        return jnp.asarray(weights.get(name, default), dtype=dtype)

    w_recon = _weight("time_l1", 1.0)
    w_commit = _weight("commit", 1.0)
    w_gan = _weight("gan", 0.1)
    w_feature = _weight("feature", 0.0)
    recon_term = w_recon * l_time
    commit_term = w_commit * commit_loss
    gan_term = w_gan * l_g
    feature_term = w_feature * l_fm
    total = recon_term + commit_term + gan_term + feature_term
    logs = {
        "total_loss": total,
        "reconstruct_loss": l_time,
        "commit_loss": commit_loss,
        "weighted_reconstruct_loss": recon_term,
        "weighted_commit_loss": commit_term,
        "gan_raw_loss": l_g,
        "weighted_gan_loss": gan_term,
        "feature_loss": l_fm,
        "weighted_feature_loss": feature_term,
    }
    return total, logs
