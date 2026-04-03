from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.training import checkpoints as flax_ckpt

try:
    import pod5
except Exception as exc:  # pragma: no cover - depends on runtime environment
    pod5 = None
    _POD5_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on runtime environment
    _POD5_IMPORT_ERROR = None

from codec.data.pod5_processing import normalize_adc_signal
from codec.models import build_audio_model
from codec.utils import discover_pod5_files

CONCAT_CHUNK_HOP = 11688


@dataclass(frozen=True)
class SourceChunkSpec:
    source_file: str
    source_read_id: str
    source_start: int
    source_stop: int
    source_length: int
    sample_rate_hz: float


def _require_pod5() -> Any:
    if pod5 is None:
        raise RuntimeError(f"pod5 library not available: {_POD5_IMPORT_ERROR}")
    return pod5


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_host_tree(tree: Any) -> Any:
    return jax.device_get(tree)


def _build_model(model_cfg: dict[str, Any] | None):
    return build_audio_model(model_cfg)


def _load_generator_variables(checkpoint_path: str | Path) -> dict[str, Any]:
    ckpt = flax_ckpt.restore_checkpoint(str(Path(checkpoint_path).resolve()), target=None)
    if not isinstance(ckpt, dict) or "gen" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint structure in {checkpoint_path}")
    gen_state = ckpt["gen"]
    params = None
    vq_vars = None
    if hasattr(gen_state, "params"):
        params = getattr(gen_state, "params")
        vq_vars = getattr(gen_state, "vq_vars", None)
    elif isinstance(gen_state, dict):
        params = gen_state.get("params")
        vq_vars = gen_state.get("vq_vars") or gen_state.get("vq")
    if params is None:
        raise RuntimeError(f"Checkpoint missing generator params: {checkpoint_path}")
    return {
        "params": freeze(_to_host_tree(params)),
        "vq": freeze(_to_host_tree(vq_vars or {})),
    }


def _resolve_data_path(path_value: str | Path | None, cfg_dir: Path) -> Path:
    if path_value is None:
        return cfg_dir.resolve()
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _merge_split_cfg(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    subdirs = merged.get("subdirs", ["."])
    if isinstance(subdirs, str):
        subdirs = [subdirs]
    merged["subdirs"] = list(subdirs or ["."])
    return merged


def _resolve_split_cfg(config: dict[str, Any], split_name: str) -> dict[str, Any]:
    cfg_dir = Path(config.get("_config_dir", ".")).resolve()
    data_cfg = dict(config.get("data") or {})
    if split_name not in data_cfg:
        raise ValueError(f"Split {split_name!r} not found in config.data")
    base_cfg = {
        "type": data_cfg.get("type", "pod5"),
        "root": data_cfg.get("root"),
        "subdirs": data_cfg.get("subdirs", ["."]),
        "segment_sec": float(data_cfg.get("segment_sec", 1.0)),
        "segment_samples": data_cfg.get("segment_samples"),
        "segment_hop_samples": data_cfg.get("segment_hop_samples"),
        "sample_rate": float(data_cfg.get("sample_rate", 5000.0)),
    }
    split_cfg = _merge_split_cfg(base_cfg, data_cfg.get(split_name))
    split_cfg["root"] = str(_resolve_data_path(split_cfg.get("root"), cfg_dir))
    return split_cfg


def _resolve_split_files(config: dict[str, Any], split_name: str) -> list[Path]:
    split_cfg = _resolve_split_cfg(config, split_name)
    explicit_files = split_cfg.get("files")
    if explicit_files:
        files = [Path(f).expanduser().resolve() for f in explicit_files]
    else:
        if split_cfg.get("type", "pod5") != "pod5":
            raise ValueError(f"Unsupported data.type={split_cfg.get('type')!r}")
        files = discover_pod5_files(Path(split_cfg["root"]), split_cfg.get("subdirs", ["."]))
    if not files:
        raise FileNotFoundError(f"No POD5 files found for split {split_name!r}")
    return files


def _resolve_segment_samples(config: dict[str, Any], split_name: str) -> int:
    split_cfg = _resolve_split_cfg(config, split_name)
    raw_value = split_cfg.get("segment_samples")
    if raw_value not in (None, ""):
        value = int(raw_value)
        if value > 0:
            return value
    sample_rate = float(split_cfg.get("sample_rate", 5000.0))
    segment_sec = float(split_cfg.get("segment_sec", 1.0))
    value = int(round(segment_sec * sample_rate))
    if value <= 0:
        raise ValueError(f"Invalid segment length for split {split_name!r}")
    return value


def _resolve_segment_hop_samples(config: dict[str, Any], split_name: str) -> int:
    split_cfg = _resolve_split_cfg(config, split_name)
    raw_value = split_cfg.get("segment_hop_samples")
    if raw_value not in (None, ""):
        value = int(raw_value)
        if value > 0:
            return value
    return _resolve_segment_samples(config, split_name)


def _iter_source_specs(
    *,
    files: Sequence[str | Path],
    chunk_size: int,
    chunk_hop: int,
    sample_rate_hz: float,
    target_chunks: int,
    chunks_per_step: int,
) -> tuple[list[SourceChunkSpec], np.ndarray, list[str]]:
    del chunks_per_step
    pod5_module = _require_pod5()
    chunk_size = max(1, int(chunk_size))
    chunk_hop = max(1, int(chunk_hop))
    target_chunks = max(1, int(target_chunks))
    warnings: list[str] = []
    specs: list[SourceChunkSpec] = []
    chunks: list[np.ndarray] = []

    for file_path_raw in files:
        file_path = Path(file_path_raw).expanduser().resolve()
        try:
            with pod5_module.Reader(str(file_path)) as reader:
                for record in reader.reads():
                    read_id = str(getattr(record, "read_id", "")) or f"{file_path.name}:{len(specs)}"
                    raw_signal = np.asarray(record.signal, dtype=np.int16)
                    if raw_signal.size < chunk_size:
                        continue
                    try:
                        normalized, _, _ = normalize_adc_signal(raw_signal, getattr(record, "calibration", None))
                    except Exception as exc:
                        warnings.append(f"skip {file_path.name}:{read_id} ({exc})")
                        continue
                    last_start = int(normalized.shape[0]) - chunk_size
                    for start in range(0, last_start + 1, chunk_hop):
                        stop = start + chunk_size
                        specs.append(
                            SourceChunkSpec(
                                source_file=str(file_path),
                                source_read_id=read_id,
                                source_start=start,
                                source_stop=stop,
                                source_length=int(raw_signal.size),
                                sample_rate_hz=float(sample_rate_hz),
                            )
                        )
                        chunks.append(np.asarray(normalized[start:stop], dtype=np.float32))
                        if len(chunks) >= target_chunks:
                            break
                    if len(chunks) >= target_chunks:
                        break
        except Exception as exc:
            warnings.append(f"failed to scan {file_path}: {exc}")
        if len(chunks) >= target_chunks:
            break

    if not chunks:
        raise RuntimeError("Unable to extract any normalized chunks from the requested POD5 sources.")

    if len(chunks) < target_chunks:
        original_specs = list(specs)
        original_chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        warnings.append(
            f"Only found {len(chunks)} chunks; repeating cached chunks to reach requested {target_chunks}."
        )
        repeat_idx = 0
        while len(chunks) < target_chunks:
            specs.append(original_specs[repeat_idx % len(original_specs)])
            chunks.append(np.array(original_chunks[repeat_idx % len(original_chunks)], copy=True))
            repeat_idx += 1

    stacked = np.stack(chunks[:target_chunks], axis=0).astype(np.float32)
    return specs[:target_chunks], stacked, warnings
