from __future__ import annotations

import numpy as np

_MAD_SCALE = 1.4826


def robust_scale_with_stats(signal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """Normalize 1D signal using median/MAD and return stats for inversion."""
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    x = signal.astype(np.float32, copy=False)
    if x.size == 0:
        return x, 0.0, 1.0
    median = np.median(x).astype(np.float32)
    deviations = np.abs(x - median)
    mad = np.median(deviations).astype(np.float32)
    scale = mad * np.float32(_MAD_SCALE)
    if not np.isfinite(scale) or scale < eps:
        scale = np.float32(eps)
    normalized = ((x - median) / scale).astype(np.float32, copy=False)
    return normalized, float(median), float(scale)


def robust_scale(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize 1D signal using median/MAD for robust center/scale."""
    normalized, _, _ = robust_scale_with_stats(signal, eps)
    return normalized
