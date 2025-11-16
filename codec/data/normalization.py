from __future__ import annotations

import numpy as np

_MAD_SCALE = 1.4826


def robust_scale(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize 1D signal using median/MAD for robust center/scale."""
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    x = signal.astype(np.float32, copy=False)
    if x.size == 0:
        return x
    median = np.median(x).astype(np.float32)
    deviations = np.abs(x - median)
    mad = np.median(deviations).astype(np.float32)
    scale = mad * np.float32(_MAD_SCALE)
    if not np.isfinite(scale) or scale < eps:
        scale = np.float32(eps)
    return ((x - median) / scale).astype(np.float32, copy=False)
