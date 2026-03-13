from __future__ import annotations

import numpy as np


def standardize_with_stats(signal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """Standardize a 1D signal using per-signal mean/std statistics."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x, 0.0, 0.0
    mean = float(np.mean(x))
    if not np.isfinite(mean):
        mean = 0.0
    centered = x - mean
    std = float(np.std(centered))
    if not np.isfinite(std):
        std = 0.0
    if std < eps:
        normalized = np.zeros_like(x, dtype=np.float32)
    else:
        normalized = np.asarray(centered / std, dtype=np.float32)
    return normalized, mean, std


def standardize(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convenience wrapper for per-signal standardization."""
    normalized, _, _ = standardize_with_stats(signal, eps)
    return normalized
