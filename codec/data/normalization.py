from __future__ import annotations

import numpy as np


def minmax_scale_with_stats(signal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """Normalize a 1D signal to [-1, 1] using per-read min/max statistics."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x, 0.0, 0.0
    data_min = float(np.min(x))
    data_max = float(np.max(x))
    if not np.isfinite(data_min):
        data_min = 0.0
    if not np.isfinite(data_max):
        data_max = 0.0
    data_range = data_max - data_min
    if not np.isfinite(data_range):
        data_range = 0.0
    if data_range < eps:
        normalized = np.zeros_like(x, dtype=np.float32)
    else:
        normalized = np.asarray(((x - data_min) / data_range) * 2.0 - 1.0, dtype=np.float32)
    return normalized, data_min, data_max


def minmax_scale(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convenience wrapper for min-max scaling to [-1, 1]."""
    normalized, _, _ = minmax_scale_with_stats(signal, eps)
    return normalized

