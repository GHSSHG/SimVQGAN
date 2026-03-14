from __future__ import annotations

import numpy as np


def normalize_to_pm1_with_stats(signal: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """Normalize a 1D signal to [-1, 1] using per-signal center/half-range."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x, 0.0, 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min):
        x_min = 0.0
    if not np.isfinite(x_max):
        x_max = x_min
    center = 0.5 * (x_min + x_max)
    half_range = 0.5 * (x_max - x_min)
    if not np.isfinite(center):
        center = 0.0
    if not np.isfinite(half_range):
        half_range = 0.0
    if half_range < eps:
        normalized = np.zeros_like(x, dtype=np.float32)
    else:
        normalized = np.asarray((x - center) / half_range, dtype=np.float32)
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized, center, half_range


def normalize_to_pm1(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convenience wrapper for per-signal [-1, 1] normalization."""
    normalized, _, _ = normalize_to_pm1_with_stats(signal, eps)
    return normalized
