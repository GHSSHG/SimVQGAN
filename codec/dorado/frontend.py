from __future__ import annotations

import jax
import jax.numpy as jnp

from .encoder import (
    DoradoPerceptualState,
    extract_dorado_conv_features,
    extract_dorado_features,
    load_dorado_perceptual_state,
    prepare_pa_for_dorado,
)


def _normalize_conv_feature_map(x: jnp.ndarray, *, eps: float = 1e-5) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    mean = jnp.mean(x, axis=1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=1, keepdims=True)
    inv_scale = jax.lax.rsqrt(var + jnp.asarray(eps * eps, dtype=x.dtype))
    return centered * inv_scale


def _rms_norm_tokens(x: jnp.ndarray, *, eps: float = 1e-5) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + jnp.asarray(eps * eps, dtype=x.dtype))
    return x / rms


def _l2_normalize(x: jnp.ndarray, *, eps: float = 1e-5) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    denom = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + jnp.asarray(eps * eps, dtype=x.dtype))
    return x / denom


def _semantic_token_cosine_loss(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    y_norm = _l2_normalize(_rms_norm_tokens(y))
    y_hat_norm = _l2_normalize(_rms_norm_tokens(y_hat))
    cosine = jnp.sum(y_norm * y_hat_norm, axis=-1)
    cosine = jnp.clip(cosine, -1.0, 1.0)
    return jnp.mean(1.0 - cosine)


def dorado_loss_scale(step: jnp.ndarray, state: DoradoPerceptualState) -> jnp.ndarray:
    step_f = jnp.asarray(step, dtype=jnp.float32)
    start = jnp.asarray(float(state.warmup_start), dtype=jnp.float32)
    warm_steps = jnp.asarray(float(state.warmup_steps), dtype=jnp.float32)
    ramp = jnp.where(
        step_f < start,
        0.0,
        jnp.where(warm_steps <= 0.0, 1.0, jnp.clip((step_f - start + 1.0) / warm_steps, 0.0, 1.0)),
    )
    return jnp.asarray(float(state.loss_weight), dtype=jnp.float32) * ramp


def compute_dorado_perceptual_loss(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    pa_center: jnp.ndarray,
    pa_half_range: jnp.ndarray,
    state: DoradoPerceptualState,
    step: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    y_dorado = prepare_pa_for_dorado(y, pa_center, pa_half_range, state)
    y_hat_dorado = prepare_pa_for_dorado(y_hat, pa_center, pa_half_range, state)
    y_feats = tuple(jax.lax.stop_gradient(feat) for feat in extract_dorado_features(y_dorado, state))
    y_hat_feats = extract_dorado_features(y_hat_dorado, state)

    scale = dorado_loss_scale(step, state)
    raw_total = jnp.asarray(0.0, dtype=jnp.float32)
    weighted_total = jnp.asarray(0.0, dtype=jnp.float32)
    logs: dict[str, jnp.ndarray] = {"dorado_perceptual_scale": scale}

    for layer_name, layer_weight, feat_y, feat_y_hat in zip(
        state.layer_names,
        state.layer_weights,
        y_feats,
        y_hat_feats,
    ):
        if layer_name.startswith("conv"):
            raw = jnp.mean(jnp.abs(_normalize_conv_feature_map(feat_y) - _normalize_conv_feature_map(feat_y_hat)))
        else:
            raw = _semantic_token_cosine_loss(feat_y, feat_y_hat)
        layer_weight_arr = jnp.asarray(float(layer_weight), dtype=jnp.float32)
        raw_total = raw_total + (layer_weight_arr * raw)
        weighted = scale * layer_weight_arr * raw
        weighted_total = weighted_total + weighted
        logs[f"dorado_{layer_name}_loss_raw"] = raw
        logs[f"dorado_{layer_name}_loss"] = weighted

    logs["dorado_perceptual_loss_raw"] = raw_total
    logs["dorado_perceptual_loss"] = weighted_total
    return weighted_total, logs


__all__ = [
    "DoradoPerceptualState",
    "compute_dorado_perceptual_loss",
    "dorado_loss_scale",
    "extract_dorado_conv_features",
    "extract_dorado_features",
    "load_dorado_perceptual_state",
    "prepare_pa_for_dorado",
]
