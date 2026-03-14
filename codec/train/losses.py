from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from ..dorado import DoradoPerceptualState
from ..dorado.frontend import compute_dorado_perceptual_loss


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


def ms_stft_logmag_l1_loss(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    *,
    scales: Sequence[tuple[int, int, int]],
) -> jnp.ndarray:
    losses = [
        stft_logmag_l1_loss(
            y,
            y_hat,
            n_fft=int(n_fft),
            win_length=int(win_length),
            hop_length=int(hop_length),
        )
        for n_fft, win_length, hop_length in scales
    ]
    if not losses:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(losses))


def _stft_scale_labels(num_scales: int) -> tuple[str, ...]:
    if num_scales == 3:
        return ("small", "medium", "large")
    if num_scales == 1:
        return ("stft",)
    return tuple(f"scale_{idx}" for idx in range(num_scales))


def compute_reconstruction_losses(
    *,
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    weights: Dict[str, float],
    stft_loss_scales: Sequence[tuple[int, int, int]] = ((256, 256, 64),),
    pa_center: jnp.ndarray | None = None,
    pa_half_range: jnp.ndarray | None = None,
    step: jnp.ndarray | None = None,
    dorado_perceptual_state: DoradoPerceptualState | None = None,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    l_time = l1_time_loss(y, y_hat)
    l_diff1 = l1_diff_loss(y, y_hat, order=1)
    l_diff2 = l1_diff_loss(y, y_hat, order=2)
    dtype = l_time.dtype

    def _weight(name: str, default: float) -> jnp.ndarray:
        return jnp.asarray(weights.get(name, default), dtype=dtype)

    w_time = _weight("time_l1", 1.0)
    w_diff1 = _weight("diff1_l1", 0.0)
    w_diff2 = _weight("diff2_l1", 0.0)
    legacy_stft_weight = float(weights.get("stft_logmag_l1", 0.0))

    time_term = w_time * l_time
    diff1_term = w_diff1 * l_diff1
    diff2_term = w_diff2 * l_diff2
    reconstruct = time_term + diff1_term + diff2_term
    logs = {
        "reconstruct_loss": reconstruct,
        "time_l1_loss_raw": l_time,
        "time_l1_loss": time_term,
        "diff1_loss_raw": l_diff1,
        "diff1_loss": diff1_term,
        "diff2_loss_raw": l_diff2,
        "diff2_loss": diff2_term,
    }

    scale_labels = _stft_scale_labels(len(stft_loss_scales))
    if len(scale_labels) != len(stft_loss_scales):
        raise ValueError("STFT scale labels must match the number of STFT scales.")
    for label, (n_fft, win_length, hop_length) in zip(scale_labels, stft_loss_scales):
        raw_loss = stft_logmag_l1_loss(
            y,
            y_hat,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
        )
        weight = _weight(f"{label}_stft_logmag_l1", legacy_stft_weight)
        weighted_loss = weight * raw_loss
        reconstruct = reconstruct + weighted_loss
        logs[f"{label}_stft_logmag_loss_raw"] = raw_loss
        logs[f"{label}_stft_logmag_loss"] = weighted_loss

    if (
        dorado_perceptual_state is not None
        and pa_center is not None
        and pa_half_range is not None
        and step is not None
    ):
        dorado_loss, dorado_logs = compute_dorado_perceptual_loss(
            y=y,
            y_hat=y_hat,
            pa_center=pa_center,
            pa_half_range=pa_half_range,
            state=dorado_perceptual_state,
            step=step,
        )
        reconstruct = reconstruct + dorado_loss
        logs.update(dorado_logs)

    logs["reconstruct_loss"] = reconstruct
    return reconstruct, logs
