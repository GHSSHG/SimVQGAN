from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp


def _as_float_signal(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _ensure_batch_time(x: jnp.ndarray) -> jnp.ndarray:
    x = _as_float_signal(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(f"Expected a (B,T) or compatible signal tensor, got {x.shape}")
    return x


def _finite_difference(x: jnp.ndarray, *, order: int) -> jnp.ndarray:
    if order < 0:
        raise ValueError(f"Difference order must be >= 0, got {order}")
    diff = x
    for _ in range(order):
        if int(diff.shape[-1]) <= 1:
            return jnp.zeros(diff.shape[:-1] + (1,), dtype=diff.dtype)
        diff = diff[..., 1:] - diff[..., :-1]
    return diff


def _hann_window(length: int, *, dtype: jnp.dtype) -> jnp.ndarray:
    if length <= 1:
        return jnp.ones((max(1, int(length)),), dtype=dtype)
    n = jnp.arange(length, dtype=dtype)
    return 0.5 - 0.5 * jnp.cos((2.0 * jnp.pi * n) / (length - 1))


def _frame_signal(x: jnp.ndarray, *, frame_length: int, hop_length: int) -> jnp.ndarray:
    total_length = int(x.shape[-1])
    if total_length < frame_length:
        x = jnp.pad(x, ((0, 0), (0, frame_length - total_length)))
        total_length = frame_length
    num_frames = 1 + max(0, (total_length - frame_length) // hop_length)
    frames = [
        jax.lax.dynamic_slice_in_dim(x, i * hop_length, frame_length, axis=-1)
        for i in range(num_frames)
    ]
    return jnp.stack(frames, axis=1)


def _stft_logmag(x: jnp.ndarray, *, n_fft: int, hop_length: int, win_length: int) -> jnp.ndarray:
    signal = _ensure_batch_time(x)
    fft_size = max(1, int(n_fft))
    frame_length = max(1, min(int(win_length), fft_size))
    hop = max(1, int(hop_length))
    window = _hann_window(frame_length, dtype=signal.dtype)
    frames = _frame_signal(signal, frame_length=frame_length, hop_length=hop)
    windowed = frames * window[None, None, :]
    spec = jnp.fft.rfft(windowed, n=fft_size, axis=-1)
    return jnp.log1p(jnp.abs(spec))


def l1_time_loss(y: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    return jnp.mean(jnp.abs(y - y_hat))


def l1_diff_loss(y: jnp.ndarray, y_hat: jnp.ndarray, *, order: int) -> jnp.ndarray:
    y = _ensure_batch_time(y)
    y_hat = _ensure_batch_time(y_hat)
    y_diff = _finite_difference(y, order=order)
    y_hat_diff = _finite_difference(y_hat, order=order)
    return jnp.mean(jnp.abs(y_diff - y_hat_diff))


def stft_logmag_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: int = 256,
) -> jnp.ndarray:
    y_logmag = _stft_logmag(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    y_hat_logmag = _stft_logmag(y_hat, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return jnp.mean(jnp.abs(y_logmag - y_hat_logmag))


def map_mean(x: jnp.ndarray) -> jnp.ndarray:
    x = _as_float_signal(x)
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def hinge_g_loss(fake_map: jnp.ndarray) -> jnp.ndarray:
    fake_map = _as_float_signal(fake_map)
    return -map_mean(fake_map).mean()


def hinge_d_loss(real_map: jnp.ndarray, fake_map: jnp.ndarray) -> jnp.ndarray:
    real_map = _as_float_signal(real_map)
    fake_map = _as_float_signal(fake_map)
    real_term = jnp.mean(jnp.maximum(0.0, 1.0 - real_map))
    fake_term = jnp.mean(jnp.maximum(0.0, 1.0 + fake_map))
    return real_term + fake_term


def feature_matching_loss(real_feats: Sequence[jnp.ndarray], fake_feats: Sequence[jnp.ndarray]) -> jnp.ndarray:
    losses = [
        jnp.mean(jnp.abs(_as_float_signal(r) - _as_float_signal(f)))
        for r, f in zip(real_feats, fake_feats)
    ]
    if not losses:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(losses))


def compute_reconstruction_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    weights: Dict[str, float],
    stft_n_fft: int = 256,
    stft_hop_length: int = 64,
    stft_win_length: int = 256,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    l_time = l1_time_loss(y, y_hat)
    l_diff1 = l1_diff_loss(y, y_hat, order=1)
    l_diff2 = l1_diff_loss(y, y_hat, order=2)
    l_stft = stft_logmag_l1_loss(
        y,
        y_hat,
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
    )
    dtype = l_time.dtype

    def _weight(name: str, default: float) -> jnp.ndarray:
        return jnp.asarray(weights.get(name, default), dtype=dtype)

    w_time = _weight("time_l1", 1.0)
    w_diff1 = _weight("diff1_l1", 0.0)
    w_diff2 = _weight("diff2_l1", 0.0)
    w_stft = _weight("stft_logmag_l1", 0.0)

    reconstruct_raw = l_time + l_diff1 + l_diff2 + l_stft
    time_term = w_time * l_time
    diff1_term = w_diff1 * l_diff1
    diff2_term = w_diff2 * l_diff2
    stft_term = w_stft * l_stft
    reconstruct = time_term + diff1_term + diff2_term + stft_term
    logs = {
        "reconstruct_loss_raw": reconstruct_raw,
        "reconstruct_loss": reconstruct,
        "time_l1_loss_raw": l_time,
        "time_l1_loss": time_term,
        "diff1_loss_raw": l_diff1,
        "diff1_loss": diff1_term,
        "diff2_loss_raw": l_diff2,
        "diff2_loss": diff2_term,
        "stft_logmag_loss_raw": l_stft,
        "stft_logmag_loss": stft_term,
    }
    return reconstruct, logs


def compute_generator_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    fake_map: jnp.ndarray,
    real_feats: Sequence[jnp.ndarray],
    fake_feats: Sequence[jnp.ndarray],
    weights: Dict[str, float],
    stft_n_fft: int = 256,
    stft_hop_length: int = 64,
    stft_win_length: int = 256,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    reconstruct, logs = compute_reconstruction_losses(
        y=y,
        y_hat=y_hat,
        weights=weights,
        stft_n_fft=stft_n_fft,
        stft_hop_length=stft_hop_length,
        stft_win_length=stft_win_length,
    )
    l_g = hinge_g_loss(fake_map)
    l_fm = feature_matching_loss(real_feats, fake_feats)
    dtype = reconstruct.dtype

    def _weight(name: str, default: float) -> jnp.ndarray:
        return jnp.asarray(weights.get(name, default), dtype=dtype)

    w_gan = _weight("gan", 0.1)
    w_feature = _weight("feature", 0.0)
    gan_term = w_gan * l_g
    feature_term = w_feature * l_fm
    total = reconstruct + gan_term + feature_term
    logs = dict(logs)
    logs.update(
        {
            "total_loss": total,
            "gan_loss_raw": l_g,
            "gan_loss": gan_term,
            "feature_loss_raw": l_fm,
            "feature_loss": feature_term,
        }
    )
    return total, logs
