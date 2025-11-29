from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

from .normalization import minmax_scale_with_stats


class CalibrationError(RuntimeError):
    """Raised when calibration metadata is missing or invalid."""


class MissingCalibrationError(CalibrationError):
    """Raised when calibration metadata is absent."""


@dataclass(frozen=True)
class CalibrationParams:
    """Per-read calibration metadata used to convert ADC counts to picoamps."""

    offset: float = 0.0
    scale: float = 1.0

    def to_picoamps(self, adc_samples: np.ndarray) -> np.ndarray:
        # NumPy 2.0+ may raise when copy=False cannot be honored, so omit the flag.
        adc = np.asarray(adc_samples, dtype=np.float32)
        safe_scale = self.scale if np.isfinite(self.scale) and self.scale != 0.0 else 1.0
        return (adc + self.offset) * safe_scale

    def to_adc(self, pa_samples: np.ndarray) -> np.ndarray:
        pa = np.asarray(pa_samples, dtype=np.float32)
        safe_scale = self.scale if np.isfinite(self.scale) and self.scale != 0.0 else 1.0
        return (pa / safe_scale) - self.offset


@dataclass(frozen=True)
class NormalizationStats:
    """Per-read min/max statistics for reversible min-max scaling."""

    data_min: float = 0.0
    data_max: float = 0.0

    @property
    def data_range(self) -> float:
        return float(self.data_max) - float(self.data_min)


def parse_calibration(calibration_obj: Any | None) -> CalibrationParams:
    """Extract CalibrationParams from POD5 Read/Run or raise when absent."""
    if isinstance(calibration_obj, CalibrationParams):
        return calibration_obj
    if calibration_obj is None:
        raise MissingCalibrationError("missing calibration metadata")
    offset = 0.0
    scale = 1.0
    source = "unknown"
    for attr in ("offset", "calibration_offset"):
        if hasattr(calibration_obj, attr):
            offset = getattr(calibration_obj, attr)
            source = attr
            break
    for attr in ("scale", "calibration_scale"):
        if hasattr(calibration_obj, attr):
            scale = getattr(calibration_obj, attr)
            break
    try:
        offset = float(offset)
    except Exception:
        offset = 0.0
    try:
        scale = float(scale)
    except Exception:
        scale = np.nan
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        msg = f"[pod5] Invalid calibration scale '{scale}' from {source}; skipping read."
        raise CalibrationError(msg)
    return CalibrationParams(offset=offset, scale=scale)


def normalize_adc_signal(
    signal: np.ndarray,
    calibration: Any | None,
    *,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, NormalizationStats, CalibrationParams]:
    """Convert ADC signal to normalized values with reversible metadata."""
    cal = parse_calibration(calibration)
    pa = cal.to_picoamps(signal)
    norm, data_min, data_max = minmax_scale_with_stats(pa, eps=eps)
    stats = NormalizationStats(data_min=data_min, data_max=data_max)
    return norm, stats, cal


def denormalize_to_adc(
    normalized: np.ndarray,
    stats: NormalizationStats,
    calibration: CalibrationParams,
    *,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert min-max normalization to obtain (pA, ADC) arrays."""
    norm = np.asarray(normalized, dtype=np.float32)
    data_min = float(stats.data_min)
    data_max = float(stats.data_max)
    data_range = data_max - data_min
    if data_range < eps:
        pa = np.full_like(norm, data_min, dtype=np.float32)
    else:
        pa = ((norm + 1.0) * 0.5) * data_range + data_min
    adc = calibration.to_adc(pa)
    return pa, adc


def resolve_sample_rate(
    *,
    read_obj: Any,
    run_info: Any | None = None,
    configured_hz: float | None = None,
    fallback_hz: float = 5000.0,
) -> float:
    """Pick the best available sample-rate hint in Hz."""
    candidates: Iterable[Any] = (
        getattr(read_obj, "sample_rate", None),
        getattr(run_info, "sample_rate", None),
        configured_hz,
        fallback_hz,
    )
    for cand in candidates:
        if cand is None:
            continue
        try:
            value = float(cand)
        except Exception:
            continue
        if value > 0.0 and np.isfinite(value):
            return value
    return float(fallback_hz)
