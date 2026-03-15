#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codec.runtime import configure_runtime_env, enable_jax_compilation_cache

configure_runtime_env()
enable_jax_compilation_cache()

import jax
import numpy as np

try:
    import edlib
except Exception:  # pragma: no cover - optional dependency
    edlib = None

try:
    import pod5
    from pod5 import Writer
except Exception as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(f"pod5 library not available: {exc}") from exc

from codec.data.normalization import normalize_to_pm1_with_stats
from codec.data.pod5_processing import (
    CalibrationError,
    CalibrationParams,
    NormalizationStats,
    denormalize_to_adc,
    parse_calibration,
)
from valid.export_valid_recon_pod5 import (
    _build_model,
    _load_generator_variables,
    _load_json,
    _resolve_segment_samples,
    _resolve_split_files,
)

RECON_MODE_DIRECT = "direct_chunk"
RECON_MODE_OVERLAP = "overlap_add"
SUPPORTED_RECON_MODES = frozenset({RECON_MODE_DIRECT, RECON_MODE_OVERLAP})
DEFAULT_OVERLAP_HOP = 2000


@dataclass(frozen=True)
class ManifestRead:
    source_file: str
    read_id: str
    raw_length: int


@dataclass(frozen=True)
class ChunkSpec:
    read_index: int
    read_id: str
    chunk_index: int
    start: int
    stop: int
    center: float
    half_range: float


@dataclass
class ReconstructionRead:
    read_index: int
    source_file: str
    read_id: str
    raw_length: int
    trimmed_length: int
    chunk_count: int
    calibration: CalibrationParams = field(repr=False)
    template_read: Any = field(repr=False)
    trimmed_raw: np.ndarray = field(repr=False)
    trimmed_pa: np.ndarray = field(repr=False)
    chunk_starts: list[int] = field(repr=False)
    overlap_weights: list[np.ndarray] | None = field(default=None, repr=False)
    reconstructed_pa: np.ndarray | None = field(default=None, repr=False)
    reconstructed_adc: np.ndarray | None = field(default=None, repr=False)
    pa_acc: np.ndarray | None = field(default=None, repr=False)
    weight_acc: np.ndarray | None = field(default=None, repr=False)
    chunk_norm_mae: float = 0.0
    chunk_norm_rmse: float = 0.0
    pa_mae: float = 0.0
    pa_rmse: float = 0.0
    adc_mae: float = 0.0
    adc_rmse: float = 0.0


@dataclass(frozen=True)
class FastqEntry:
    seq: str
    qual: str


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _parse_args() -> argparse.Namespace:
    chunk_batch_default = _env_int("CHUNK_BATCH_SIZE")
    if chunk_batch_default is None:
        chunk_batch_default = _env_int("MICROBATCH") or 128
    source_pod5_default = os.environ.get("SOURCE_POD5")
    manifest_path_default = os.environ.get("MANIFEST_PATH")

    parser = argparse.ArgumentParser(description="Run SimVQGAN validation for a saved checkpoint.")
    parser.add_argument("--config", type=Path, default=os.environ.get("CONFIG_PATH"))
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--num-reads", type=int, default=int(os.environ.get("NUM_READS", "64")))
    parser.add_argument("--min-read-length", type=int, default=int(os.environ.get("MIN_READ_LENGTH", "12288")))
    parser.add_argument("--max-read-length", type=int, default=_env_int("MAX_READ_LENGTH"))
    parser.add_argument("--microbatch", type=int, default=None, help="Legacy alias for --chunk-batch-size.")
    parser.add_argument("--chunk-batch-size", type=int, default=chunk_batch_default)
    parser.add_argument("--data-split", type=str, default=os.environ.get("DATA_SPLIT", "valid"))
    parser.add_argument("--source-pod5", type=Path, default=(Path(source_pod5_default) if source_pod5_default else None))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=(Path(manifest_path_default) if manifest_path_default else None),
    )
    parser.add_argument("--recon-mode", type=str, default=os.environ.get("RECON_MODE", RECON_MODE_DIRECT))
    parser.add_argument("--hop-samples", type=int, default=_env_int("HOP_SAMPLES"))
    parser.add_argument("--dorado-bin", type=str, default=os.environ.get("DORADO_BIN", "dorado"))
    parser.add_argument("--dorado-model", type=str, default=os.environ.get("DORADO_MODEL"))
    parser.add_argument("--dorado-device", type=str, default=os.environ.get("DORADO_DEVICE", "cuda:0"))
    parser.add_argument("--trim-mode", type=str, default=os.environ.get("TRIM_MODE", "drop"))
    parser.add_argument("--prepare-manifest-only", action="store_true", default=_env_flag("PREPARE_MANIFEST_ONLY"))
    parser.add_argument("--skip-dorado", action="store_true", default=_env_flag("SKIP_DORADO"))
    args = parser.parse_args()

    if args.microbatch is not None:
        args.chunk_batch_size = args.microbatch
    args.chunk_batch_size = max(1, int(args.chunk_batch_size))
    args.recon_mode = str(args.recon_mode).strip().lower() or RECON_MODE_DIRECT
    if args.recon_mode not in SUPPORTED_RECON_MODES:
        raise SystemExit(f"Unsupported --recon-mode={args.recon_mode!r}; choose from {sorted(SUPPORTED_RECON_MODES)}")
    args.trim_mode = str(args.trim_mode).strip().lower() or "drop"
    if args.trim_mode not in {"drop", "pad"}:
        raise SystemExit(f"Unsupported --trim-mode={args.trim_mode!r}; choose from ['drop', 'pad']")

    if args.config is None:
        raise SystemExit("CONFIG_PATH/--config is required")
    if args.output_dir is None and not args.prepare_manifest_only:
        raise SystemExit("OUTPUT_DIR/--output-dir is required")
    if args.checkpoint is None and not args.prepare_manifest_only:
        raise SystemExit("CHECKPOINT_PATH/--checkpoint is required")
    if args.output_dir is None and args.manifest_path is None:
        raise SystemExit("Either OUTPUT_DIR/--output-dir or MANIFEST_PATH/--manifest-path is required")
    return args


def _resolve_repo_path(path_value: str | Path | None, cfg_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    raw_text = str(path_value).strip()
    if not raw_text:
        return None
    candidate = Path(raw_text).expanduser()
    has_explicit_path = candidate.is_absolute() or raw_text.startswith(".") or raw_text.startswith("~")
    has_explicit_path = has_explicit_path or any(sep in raw_text for sep in (os.sep, "/", "\\"))
    if not has_explicit_path and not candidate.exists():
        return candidate
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _is_executable_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_dorado_bin(
    dorado_bin_value: str | Path | None,
    *,
    cfg_dir: Path,
    dorado_model: Path | None,
) -> Path:
    raw_value = "" if dorado_bin_value is None else str(dorado_bin_value).strip()
    if raw_value:
        explicit = _resolve_repo_path(raw_value, cfg_dir)
        if explicit is not None and _is_executable_file(explicit):
            return explicit
        if explicit is None:
            which_match = shutil.which(raw_value)
            if which_match:
                return Path(which_match).resolve()

    candidate_paths: list[Path] = []
    if dorado_model is not None:
        model_dir = dorado_model.resolve()
        search_roots = [model_dir]
        search_roots.extend(model_dir.parents)
        for root in search_roots:
            candidate_paths.append(root / "dorado")
            candidate_paths.append(root / "bin" / "dorado")
            candidate_paths.extend(sorted(root.glob("dorado-*/bin/dorado")))
            if root.name == "models":
                parent = root.parent
                candidate_paths.append(parent / "bin" / "dorado")
                candidate_paths.extend(sorted(parent.glob("dorado-*/bin/dorado")))

    default_roots = [
        Path("~/Download/dorado").expanduser(),
        Path("~/dorado").expanduser(),
    ]
    for root in default_roots:
        candidate_paths.append(root / "dorado")
        candidate_paths.append(root / "bin" / "dorado")
        candidate_paths.extend(sorted(root.glob("dorado-*/bin/dorado")))

    for candidate in _dedupe_paths(path.resolve() for path in candidate_paths if _is_executable_file(path)):
        return candidate

    fallback = shutil.which("dorado")
    if fallback:
        return Path(fallback).resolve()

    searched = []
    if raw_value:
        searched.append(raw_value)
    if dorado_model is not None:
        searched.append(str(dorado_model))
    raise FileNotFoundError(
        "Unable to locate the Dorado executable. "
        "Set DORADO_BIN/--dorado-bin or install Dorado into a standard location. "
        f"Searched from: {searched}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _progress_markers(total: int) -> list[int]:
    total = max(0, int(total))
    if total <= 0:
        return []
    markers = {1, total}
    for pct in (10, 25, 50, 75, 90):
        markers.add(max(1, int(round(total * pct / 100.0))))
    return sorted(markers)


def _summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _sanitize_tag(text: str) -> str:
    cleaned = [ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text)]
    return "".join(cleaned).strip("_") or "data"


def _resolve_recon_hop(recon_mode: str, chunk_size: int, hop_override: int | None) -> int:
    if hop_override is None:
        hop_samples = int(chunk_size) if recon_mode == RECON_MODE_DIRECT else int(DEFAULT_OVERLAP_HOP)
    else:
        hop_samples = int(hop_override)
    if hop_samples <= 0:
        raise ValueError(f"Validation chunk hop must be positive, got {hop_samples}.")
    if hop_samples > int(chunk_size):
        raise ValueError(
            f"Validation chunk hop ({hop_samples}) cannot exceed chunk size ({int(chunk_size)})."
        )
    return hop_samples


def _trimmed_signal_length(
    signal_length: int,
    chunk_size: int,
    hop_size: int,
    trim_mode: str,
) -> int | None:
    n = max(0, int(signal_length))
    chunk_size = max(1, int(chunk_size))
    hop_size = max(1, int(hop_size))
    if hop_size > chunk_size:
        raise ValueError(f"hop_size ({hop_size}) cannot exceed chunk_size ({chunk_size}).")
    if n < chunk_size:
        if trim_mode == "pad" and n > 0:
            return chunk_size
        return None
    if trim_mode == "drop":
        step_count = 1 + ((n - chunk_size) // hop_size)
        return chunk_size + ((step_count - 1) * hop_size)
    if trim_mode == "pad":
        extra = n - chunk_size
        step_count = 1 + (extra // hop_size)
        if (extra % hop_size) != 0:
            step_count += 1
        return chunk_size + ((step_count - 1) * hop_size)
    raise ValueError(f"Unsupported trim_mode={trim_mode!r}")


def _window_starts(total_samples: int, chunk_size: int, hop_size: int) -> list[int]:
    trimmed_length = _trimmed_signal_length(total_samples, chunk_size, hop_size, "drop")
    if trimmed_length is None:
        return []
    last_start = trimmed_length - int(chunk_size)
    return list(range(0, last_start + 1, int(hop_size)))


def _chunk_crossfade_weights(
    window_index: int,
    window_count: int,
    chunk_size: int,
    hop_size: int,
) -> np.ndarray:
    weights = np.ones((int(chunk_size),), dtype=np.float32)
    overlap_size = int(chunk_size) - int(hop_size)
    if overlap_size <= 0 or window_count <= 1:
        return weights
    fade_in = np.arange(overlap_size, dtype=np.float32) / float(overlap_size)
    fade_out = 1.0 - fade_in
    if window_index > 0:
        weights[:overlap_size] = fade_in
    if (window_index + 1) < window_count:
        weights[-overlap_size:] = fade_out
    return weights


def _trim_signal(signal: np.ndarray, segment_samples: int, hop_samples: int, trim_mode: str) -> np.ndarray | None:
    raw = np.asarray(signal, dtype=np.int16).reshape(-1)
    n = int(raw.size)
    trimmed_length = _trimmed_signal_length(n, segment_samples, hop_samples, trim_mode)
    if trimmed_length is None:
        return None
    if trimmed_length <= n:
        return raw[:trimmed_length]
    if n <= 0:
        return None
    pad = np.full((trimmed_length - n,), raw[-1], dtype=raw.dtype)
    return np.concatenate([raw, pad])


def _resolve_input_files(config: dict[str, Any], split: str, source_pod5: Path | None) -> list[Path]:
    if source_pod5 is not None:
        resolved = source_pod5.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Source POD5 not found: {resolved}")
        return [resolved]
    return _resolve_split_files(config, split)


def _manifest_path(
    eval_root: Path,
    split: str,
    files: Sequence[Path],
    num_reads: int,
    min_read_length: int,
    max_read_length: int | None,
) -> Path:
    source_tag = _sanitize_tag(Path(files[0]).stem if len(files) == 1 else split)
    parts = [f"selected_reads_{source_tag}_{split}", f"n{int(num_reads)}", f"min{int(min_read_length)}"]
    if max_read_length is not None:
        parts.append(f"max{int(max_read_length)}")
    return eval_root / ("_".join(parts) + ".json")


def _load_or_create_manifest(
    *,
    manifest_path: Path,
    split: str,
    files: list[Path],
    num_reads: int,
    min_read_length: int,
    max_read_length: int | None,
) -> tuple[list[ManifestRead], Path, list[str]]:
    current_file_set = {str(path.resolve()) for path in files}
    if manifest_path.exists():
        payload = _load_json(manifest_path)
        payload_min = payload.get("min_read_length")
        payload_max = payload.get("max_read_length")
        if payload_min not in (None, int(min_read_length)):
            raise RuntimeError(
                f"Manifest {manifest_path} was created for min_read_length={payload_min}, requested {min_read_length}."
            )
        if payload_max != (None if max_read_length is None else int(max_read_length)):
            raise RuntimeError(
                f"Manifest {manifest_path} was created for max_read_length={payload_max}, requested {max_read_length}."
            )
        items = [ManifestRead(**item) for item in payload.get("selected_reads", [])]
        for item in items:
            if str(Path(item.source_file).resolve()) not in current_file_set:
                raise RuntimeError(
                    f"Manifest {manifest_path} includes source file outside current input set: {item.source_file}"
                )
        return items, manifest_path, list(payload.get("warnings", []))

    warnings: list[str] = []
    selected: list[ManifestRead] = []
    markers = _progress_markers(len(files))
    print(f"[select] scanning {len(files)} POD5 files for split={split}", flush=True)
    for file_idx, file_path in enumerate(files, start=1):
        if markers and file_idx >= markers[0]:
            print(f"[select] file progress {file_idx}/{len(files)}", flush=True)
            markers.pop(0)
        with pod5.Reader(str(file_path)) as reader:
            for record in reader.reads():
                read_id = str(getattr(record, "read_id", ""))
                raw = np.asarray(record.signal, dtype=np.int16)
                raw_length = int(raw.size)
                if raw_length < int(min_read_length):
                    continue
                if max_read_length is not None and raw_length > int(max_read_length):
                    continue
                try:
                    parse_calibration(getattr(record, "calibration", None))
                except CalibrationError as exc:
                    if len(warnings) < 64:
                        warnings.append(f"skip {file_path.name}:{read_id} ({exc})")
                    continue
                selected.append(
                    ManifestRead(
                        source_file=str(file_path),
                        read_id=read_id,
                        raw_length=raw_length,
                    )
                )
                if len(selected) >= num_reads:
                    break
        if len(selected) >= num_reads:
            break

    if not selected:
        raise RuntimeError(
            f"No reads found for split={split!r} with min_read_length={min_read_length} and max_read_length={max_read_length}."
        )
    if len(selected) < num_reads:
        warnings.append(
            f"Requested {num_reads} reads but only found {len(selected)} matching reads; proceeding with the smaller set."
        )

    payload = {
        "split": split,
        "requested_read_count": int(num_reads),
        "selected_read_count": len(selected),
        "min_read_length": int(min_read_length),
        "max_read_length": None if max_read_length is None else int(max_read_length),
        "source_files": [str(path.resolve()) for path in files],
        "selected_reads": [item.__dict__ for item in selected],
        "warnings": warnings,
    }
    _write_json(manifest_path, payload)
    return selected, manifest_path, warnings


def _fetch_records_by_id(reader: Any, ordered_ids: Sequence[str]) -> dict[str, Any]:
    target_ids = [str(read_id) for read_id in ordered_ids]
    records: dict[str, Any] = {}
    try:
        iterator = reader.reads(selection=target_ids)
        for record in iterator:
            records[str(getattr(record, "read_id", ""))] = record
    except Exception:
        target_set = set(target_ids)
        for record in reader.reads():
            read_id = str(getattr(record, "read_id", ""))
            if read_id in target_set:
                records[read_id] = record
                if len(records) >= len(target_set):
                    break
    missing = [read_id for read_id in target_ids if read_id not in records]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested reads; first missing read_id={missing[0]}")
    return records


def _write_selected_real_pod5(
    *,
    manifest_reads: list[ManifestRead],
    output_path: Path,
    segment_samples: int,
    hop_samples: int,
    trim_mode: str,
) -> tuple[int, dict[str, int]]:
    grouped: dict[str, list[ManifestRead]] = {}
    for item in manifest_reads:
        grouped.setdefault(item.source_file, []).append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    expected = len(manifest_reads)
    read_lengths: dict[str, int] = {}
    written = 0
    markers = _progress_markers(expected)
    print(f"[real] writing selected POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for file_path_str, items in grouped.items():
            ordered_ids = [item.read_id for item in items]
            with pod5.Reader(str(Path(file_path_str))) as reader:
                fetched = _fetch_records_by_id(reader, ordered_ids)
                for item in items:
                    record = fetched[item.read_id]
                    trimmed = _trim_signal(
                        np.asarray(record.signal, dtype=np.int16),
                        segment_samples,
                        hop_samples,
                        trim_mode,
                    )
                    if trimmed is None:
                        raise RuntimeError(
                            f"Selected read became empty after trim: {file_path_str}:{item.read_id}"
                        )
                    read = record.to_read()
                    read.signal = np.asarray(trimmed, dtype=np.int16)
                    writer.add_read(read)
                    read_lengths[item.read_id] = int(trimmed.size)
                    written += 1
                    if markers and written >= markers[0]:
                        print(f"[real] progress {written}/{expected}", flush=True)
                        markers.pop(0)
    print(f"[real] done. kept {written}/{expected} reads", flush=True)
    return written, read_lengths


class _CheckpointReconstructor:
    def __init__(self, *, model_cfg: dict[str, Any], checkpoint_path: Path, chunk_batch_size: int):
        self.chunk_batch_size = max(1, int(chunk_batch_size))
        self.variables = _load_generator_variables(checkpoint_path)
        self.model = _build_model(model_cfg)
        self.apply_rng = jax.random.PRNGKey(0)

        @jax.jit
        def _reconstruct(batch: jax.Array) -> jax.Array:
            outputs = self.model.apply(
                self.variables,
                batch,
                train=False,
                offset=0,
                rng=self.apply_rng,
                collect_codebook_stats=False,
            )
            wave_hat = outputs["wave_hat"]
            if wave_hat.ndim == 3 and wave_hat.shape[1] == 1:
                wave_hat = wave_hat[:, 0, :]
            elif wave_hat.ndim != 2:
                wave_hat = wave_hat.reshape(wave_hat.shape[0], -1)
            return wave_hat

        self._reconstruct = _reconstruct

    def reconstruct_chunks(self, chunks: np.ndarray) -> np.ndarray:
        chunks = np.asarray(chunks, dtype=np.float32)
        if chunks.ndim != 2:
            raise ValueError(f"Expected chunk batch with shape [N, T], got {chunks.shape}")
        total_chunks, chunk_size = chunks.shape
        outputs = np.empty((total_chunks, chunk_size), dtype=np.float32)
        for batch_start in range(0, total_chunks, self.chunk_batch_size):
            batch = np.asarray(chunks[batch_start : batch_start + self.chunk_batch_size], dtype=np.float32)
            valid = int(batch.shape[0])
            if valid < self.chunk_batch_size:
                padded = np.zeros((self.chunk_batch_size, chunk_size), dtype=np.float32)
                padded[:valid] = batch
                batch = padded
            reconstructed = np.asarray(self._reconstruct(batch), dtype=np.float32)
            outputs[batch_start : batch_start + valid] = reconstructed[:valid]
        return outputs


def _load_prepared_reads(
    *,
    source_pod5: Path,
    segment_samples: int,
    hop_samples: int,
    recon_mode: str,
) -> tuple[list[ReconstructionRead], list[ChunkSpec], np.ndarray]:
    reads: list[ReconstructionRead] = []
    chunk_specs: list[ChunkSpec] = []
    chunks: list[np.ndarray] = []
    total_reads = _count_reads(source_pod5)
    markers = _progress_markers(total_reads)
    print(f"[prep] building chunk pool from {source_pod5}", flush=True)
    with pod5.Reader(str(source_pod5)) as reader:
        for read_index, record in enumerate(reader.reads(), start=0):
            if markers and (read_index + 1) >= markers[0]:
                print(f"[prep] read progress {read_index + 1}/{total_reads}", flush=True)
                markers.pop(0)
            read_id = str(getattr(record, "read_id", ""))
            trimmed_raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
            calibration = parse_calibration(getattr(record, "calibration", None))
            trimmed_pa = calibration.to_picoamps(trimmed_raw)
            starts = _window_starts(int(trimmed_raw.size), segment_samples, hop_samples)
            if not starts:
                raise RuntimeError(f"Selected POD5 read has no reconstructable chunks: {read_id}")
            overlap_weights = None
            if recon_mode == RECON_MODE_OVERLAP:
                overlap_weights = [
                    _chunk_crossfade_weights(chunk_index, len(starts), segment_samples, hop_samples)
                    for chunk_index in range(len(starts))
                ]
            state = ReconstructionRead(
                read_index=read_index,
                source_file=str(source_pod5),
                read_id=read_id,
                raw_length=int(trimmed_raw.size),
                trimmed_length=int(trimmed_raw.size),
                chunk_count=len(starts),
                calibration=calibration,
                template_read=record.to_read(),
                trimmed_raw=trimmed_raw,
                trimmed_pa=np.asarray(trimmed_pa, dtype=np.float32),
                chunk_starts=starts,
                overlap_weights=overlap_weights,
            )
            if recon_mode == RECON_MODE_DIRECT:
                state.reconstructed_pa = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
                state.reconstructed_adc = np.zeros((int(trimmed_raw.size),), dtype=np.int16)
            else:
                state.pa_acc = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
                state.weight_acc = np.zeros((int(trimmed_raw.size),), dtype=np.float32)
            reads.append(state)
            for chunk_index, start in enumerate(starts):
                stop = start + int(segment_samples)
                chunk_pa = np.asarray(trimmed_pa[start:stop], dtype=np.float32)
                normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
                chunks.append(np.asarray(normalized, dtype=np.float32))
                chunk_specs.append(
                    ChunkSpec(
                        read_index=read_index,
                        read_id=read_id,
                        chunk_index=chunk_index,
                        start=int(start),
                        stop=int(stop),
                        center=float(center),
                        half_range=float(half_range),
                    )
                )
    if not chunks:
        raise RuntimeError("Prepared chunk pool is empty.")
    print(f"[prep] done. reads={len(reads)} chunks={len(chunks)}", flush=True)
    return reads, chunk_specs, np.stack(chunks, axis=0).astype(np.float32)


def _finalize_reconstruction(
    *,
    reads: list[ReconstructionRead],
    chunk_specs: Sequence[ChunkSpec],
    chunk_inputs: np.ndarray,
    chunk_outputs: np.ndarray,
    recon_mode: str,
) -> dict[str, Any]:
    read_count = len(reads)
    norm_abs = np.zeros((read_count,), dtype=np.float64)
    norm_sq = np.zeros((read_count,), dtype=np.float64)
    norm_count = np.zeros((read_count,), dtype=np.int64)

    for chunk_index, spec in enumerate(chunk_specs):
        read_state = reads[spec.read_index]
        output_norm = np.asarray(chunk_outputs[chunk_index], dtype=np.float32)
        input_norm = np.asarray(chunk_inputs[chunk_index], dtype=np.float32)
        diff = output_norm - input_norm
        norm_abs[spec.read_index] += float(np.sum(np.abs(diff), dtype=np.float64))
        norm_sq[spec.read_index] += float(np.sum(np.square(diff), dtype=np.float64))
        norm_count[spec.read_index] += int(diff.size)

        stats = NormalizationStats(center=float(spec.center), half_range=float(spec.half_range))
        pa_chunk, adc_chunk = denormalize_to_adc(output_norm, stats, read_state.calibration)
        pa_chunk = np.asarray(pa_chunk, dtype=np.float32)
        adc_chunk = np.asarray(np.clip(np.rint(adc_chunk), -32768, 32767), dtype=np.int16)

        if recon_mode == RECON_MODE_DIRECT:
            if read_state.reconstructed_pa is None or read_state.reconstructed_adc is None:
                raise RuntimeError(f"Direct reconstruction buffers missing for read {read_state.read_id}")
            read_state.reconstructed_pa[spec.start : spec.stop] = pa_chunk
            read_state.reconstructed_adc[spec.start : spec.stop] = adc_chunk
        else:
            if read_state.pa_acc is None or read_state.weight_acc is None or read_state.overlap_weights is None:
                raise RuntimeError(f"Overlap reconstruction buffers missing for read {read_state.read_id}")
            weights = np.asarray(read_state.overlap_weights[spec.chunk_index], dtype=np.float32)
            read_state.pa_acc[spec.start : spec.stop] += pa_chunk * weights
            read_state.weight_acc[spec.start : spec.stop] += weights

    for read_index, read_state in enumerate(reads):
        if recon_mode == RECON_MODE_OVERLAP:
            if read_state.pa_acc is None or read_state.weight_acc is None:
                raise RuntimeError(f"Overlap accumulators missing for read {read_state.read_id}")
            reconstructed_pa = np.divide(
                read_state.pa_acc,
                np.where(read_state.weight_acc > 0.0, read_state.weight_acc, 1.0),
            )
            reconstructed_adc = read_state.calibration.to_adc(reconstructed_pa)
            read_state.reconstructed_pa = np.asarray(reconstructed_pa, dtype=np.float32)
            read_state.reconstructed_adc = np.asarray(
                np.clip(np.rint(reconstructed_adc), -32768, 32767),
                dtype=np.int16,
            )
        if read_state.reconstructed_pa is None or read_state.reconstructed_adc is None:
            raise RuntimeError(f"Missing reconstruction output for read {read_state.read_id}")

        count = max(1, int(norm_count[read_index]))
        read_state.chunk_norm_mae = float(norm_abs[read_index] / float(count))
        read_state.chunk_norm_rmse = float(np.sqrt(norm_sq[read_index] / float(count)))

        pa_diff = np.asarray(read_state.reconstructed_pa, dtype=np.float32) - np.asarray(read_state.trimmed_pa, dtype=np.float32)
        adc_diff = read_state.reconstructed_adc.astype(np.float32) - read_state.trimmed_raw.astype(np.float32)
        read_state.pa_mae = float(np.mean(np.abs(pa_diff), dtype=np.float64))
        read_state.pa_rmse = float(np.sqrt(np.mean(np.square(pa_diff), dtype=np.float64)))
        read_state.adc_mae = float(np.mean(np.abs(adc_diff), dtype=np.float64))
        read_state.adc_rmse = float(np.sqrt(np.mean(np.square(adc_diff), dtype=np.float64)))

    return {
        "processed_reads": len(reads),
        "total_chunk_count": int(len(chunk_specs)),
        "trimmed_length_summary": _summarize(read.trimmed_length for read in reads),
        "chunk_count_summary": _summarize(read.chunk_count for read in reads),
        "chunk_norm_mae_summary": _summarize(read.chunk_norm_mae for read in reads),
        "chunk_norm_rmse_summary": _summarize(read.chunk_norm_rmse for read in reads),
        "pa_mae_summary": _summarize(read.pa_mae for read in reads),
        "pa_rmse_summary": _summarize(read.pa_rmse for read in reads),
        "adc_mae_summary": _summarize(read.adc_mae for read in reads),
        "adc_rmse_summary": _summarize(read.adc_rmse for read in reads),
    }


def _write_generated_pod5(*, output_path: Path, reads: Sequence[ReconstructionRead]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    expected = len(reads)
    written = 0
    markers = _progress_markers(expected)
    print(f"[gen] writing reconstructed POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for index, read_state in enumerate(reads, start=1):
            if read_state.reconstructed_adc is None:
                raise RuntimeError(f"Missing reconstructed ADC for read {read_state.read_id}")
            read = read_state.template_read
            read.signal = np.asarray(read_state.reconstructed_adc, dtype=np.int16)
            writer.add_read(read)
            written += 1
            if markers and index >= markers[0]:
                print(f"[gen] progress {index}/{expected}", flush=True)
                markers.pop(0)
    print(f"[gen] done. kept {written}/{expected} reads", flush=True)
    return written


def _write_reconstruction_per_read_metrics(path: Path, reads: Sequence[ReconstructionRead]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for read in reads:
            record = {
                "source_file": read.source_file,
                "read_id": read.read_id,
                "raw_length": int(read.raw_length),
                "trimmed_length": int(read.trimmed_length),
                "chunk_count": int(read.chunk_count),
                "chunk_norm_mae": float(read.chunk_norm_mae),
                "chunk_norm_rmse": float(read.chunk_norm_rmse),
                "pa_mae": float(read.pa_mae),
                "pa_rmse": float(read.pa_rmse),
                "adc_mae": float(read.adc_mae),
                "adc_rmse": float(read.adc_rmse),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_reads(path: Path) -> int:
    with pod5.Reader(str(path)) as reader:
        return sum(1 for _ in reader.reads())


def _run_dorado(*, dorado_bin: str, dorado_model: str, pod5_path: Path, out_fastq: Path, device: str) -> None:
    out_fastq.parent.mkdir(parents=True, exist_ok=True)
    if out_fastq.exists():
        out_fastq.unlink()
    cmd = [dorado_bin, "basecaller", dorado_model, str(pod5_path), "--device", device, "--emit-fastq"]
    print(f"[dorado] {' '.join(cmd)}", flush=True)
    with out_fastq.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Dorado failed on {pod5_path}: {proc.stderr}")
    print(f"[dorado] finished {pod5_path.name}", flush=True)


def _open_fastq(path: Path):
    text_kwargs = {"encoding": "utf-8", "errors": "ignore"}
    if path.suffix == ".gz":
        return gzip.open(path, "rt", **text_kwargs)
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, "rt", **text_kwargs)
    except FileNotFoundError:
        raise
    return path.open("r", **text_kwargs)


def _read_fastq(path: Path) -> dict[str, FastqEntry]:
    records: dict[str, FastqEntry] = {}
    with _open_fastq(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            if not header.startswith("@"):
                continue
            read_id = header[1:].strip().split()[0]
            seq_parts: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.startswith("+"):
                    break
                seq_parts.append(line.strip())
            sequence = "".join(seq_parts)
            qual_parts: list[str] = []
            qual_len = 0
            while qual_len < len(sequence):
                line = handle.readline()
                if not line:
                    break
                qline = line.strip()
                qual_parts.append(qline)
                qual_len += len(qline)
            records[read_id] = FastqEntry(seq=sequence, qual="".join(qual_parts)[: len(sequence)])
    return records


def _mean_qscore(quality: str) -> float:
    if not quality:
        return 0.0
    values = np.fromiter((ord(ch) - 33 for ch in quality), dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _alignment_identity(a: str, b: str) -> tuple[float, int, int, str]:
    if not a and not b:
        return 1.0, 0, 0, "empty"
    if not a or not b:
        denom = max(len(a), len(b))
        return 0.0, denom, denom, "trivial"
    denom = max(len(a), len(b))
    if edlib is not None:
        res = edlib.align(a, b, mode="NW", task="distance")
        dist = int(res["editDistance"])
        identity = 1.0 - (float(dist) / float(denom))
        return identity, dist, denom, "edlib"
    ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    identity = float(ratio)
    approx_dist = int(round((1.0 - identity) * float(denom)))
    return identity, approx_dist, denom, "difflib"


def _compute_metrics(real_fastq: Path, generated_fastq: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    real = _read_fastq(real_fastq)
    generated = _read_fastq(generated_fastq)
    shared_ids = sorted(set(real) & set(generated))
    original_only = sorted(set(real) - set(generated))
    generated_only = sorted(set(generated) - set(real))

    per_read: list[dict[str, Any]] = []
    identities: list[float] = []
    length_deltas: list[float] = []
    qscore_deltas: list[float] = []
    exact_matches = 0
    total_weight = 0
    total_identity_weight = 0.0
    backend = "none"

    for read_id in shared_ids:
        real_entry = real[read_id]
        generated_entry = generated[read_id]
        identity, dist, denom, backend = _alignment_identity(real_entry.seq, generated_entry.seq)
        real_len = len(real_entry.seq)
        generated_len = len(generated_entry.seq)
        qscore_real = _mean_qscore(real_entry.qual)
        qscore_generated = _mean_qscore(generated_entry.qual)
        qscore_delta = qscore_generated - qscore_real
        length_delta = float(generated_len - real_len)
        if real_entry.seq == generated_entry.seq:
            exact_matches += 1
        total_weight += denom
        total_identity_weight += identity * float(denom)
        identities.append(float(identity))
        length_deltas.append(length_delta)
        qscore_deltas.append(float(qscore_delta))
        per_read.append(
            {
                "read_id": read_id,
                "real_length": real_len,
                "generated_length": generated_len,
                "identity": float(identity),
                "edit_distance": int(dist),
                "identity_weight": int(denom),
                "qscore_real": float(qscore_real),
                "qscore_generated": float(qscore_generated),
                "qscore_delta": float(qscore_delta),
                "length_delta": length_delta,
                "exact_match": bool(real_entry.seq == generated_entry.seq),
            }
        )

    shared_count = len(shared_ids)
    summary = {
        "shared_read_count": int(shared_count),
        "original_read_count": int(len(real)),
        "reconstructed_read_count": int(len(generated)),
        "original_only_read_count": int(len(original_only)),
        "reconstructed_only_read_count": int(len(generated_only)),
        "exact_match_rate": float(exact_matches / shared_count) if shared_count else 0.0,
        "length_weighted_identity": float(total_identity_weight / total_weight) if total_weight else 0.0,
        "mean_qscore_delta": float(np.mean(np.asarray(qscore_deltas, dtype=np.float64))) if qscore_deltas else 0.0,
        "mean_qscore_delta_abs": (
            float(np.mean(np.abs(np.asarray(qscore_deltas, dtype=np.float64)))) if qscore_deltas else 0.0
        ),
        "identity_summary": _summarize(identities),
        "length_delta_summary": _summarize(length_deltas),
        "qscore_delta_summary": _summarize(qscore_deltas),
        "identity_backend": backend,
    }
    return summary, per_read


def _write_per_read_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint).expanduser().resolve()
    output_dir = None if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    source_pod5 = None if args.source_pod5 is None else Path(args.source_pod5).expanduser().resolve()
    manifest_path_arg = None if args.manifest_path is None else Path(args.manifest_path).expanduser().resolve()
    split = str(args.data_split).strip().lower() or "valid"
    num_reads = max(1, int(args.num_reads))
    min_read_length = max(1, int(args.min_read_length))
    max_read_length = None if args.max_read_length is None else int(args.max_read_length)
    chunk_batch_size = max(1, int(args.chunk_batch_size))
    recon_mode = str(args.recon_mode).strip().lower() or RECON_MODE_DIRECT
    trim_mode = str(args.trim_mode).strip().lower() or "drop"

    config = _load_json(config_path)
    config["_config_dir"] = str(config_path.parent.resolve())
    segment_samples = _resolve_segment_samples(config, split)
    hop_samples = _resolve_recon_hop(recon_mode, segment_samples, args.hop_samples)
    split_files = _resolve_input_files(config, split, source_pod5)

    if manifest_path_arg is not None:
        manifest_path = manifest_path_arg
    else:
        if output_dir is None:
            raise RuntimeError("Automatic manifest path requires --output-dir")
        manifest_path = _manifest_path(output_dir.parent, split, split_files, num_reads, min_read_length, max_read_length)

    manifest_reads, manifest_path, manifest_warnings = _load_or_create_manifest(
        manifest_path=manifest_path,
        split=split,
        files=split_files,
        num_reads=num_reads,
        min_read_length=min_read_length,
        max_read_length=max_read_length,
    )

    if args.prepare_manifest_only:
        payload = {
            "status": "manifest_ready",
            "manifest_path": str(manifest_path),
            "requested_read_count": int(num_reads),
            "selected_read_count": int(len(manifest_reads)),
            "min_read_length": int(min_read_length),
            "max_read_length": None if max_read_length is None else int(max_read_length),
            "source_file_count": len(split_files),
            "source_files": [str(path) for path in split_files],
            "warnings": manifest_warnings,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    if checkpoint_path is None:
        raise RuntimeError("Checkpoint path is required for reconstruction")
    if output_dir is None:
        raise RuntimeError("Output directory is required for reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    pod5_dir = output_dir / "pod5"
    fastq_dir = output_dir / "fastq"
    metrics_dir = output_dir / "metrics"
    pod5_dir.mkdir(parents=True, exist_ok=True)
    fastq_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    real_pod5_path = pod5_dir / "original_selected.pod5"
    real_written, trimmed_lengths = _write_selected_real_pod5(
        manifest_reads=manifest_reads,
        output_path=real_pod5_path,
        segment_samples=segment_samples,
        hop_samples=hop_samples,
        trim_mode=trim_mode,
    )
    if real_written <= 0:
        raise RuntimeError("Selected validation POD5 is empty after trimming.")

    prepared_reads, chunk_specs, chunk_inputs = _load_prepared_reads(
        source_pod5=real_pod5_path,
        segment_samples=segment_samples,
        hop_samples=hop_samples,
        recon_mode=recon_mode,
    )
    reconstructor = _CheckpointReconstructor(
        model_cfg=dict(config.get("model") or {}),
        checkpoint_path=checkpoint_path,
        chunk_batch_size=chunk_batch_size,
    )
    chunk_outputs = reconstructor.reconstruct_chunks(chunk_inputs)
    reconstruction_summary = _finalize_reconstruction(
        reads=prepared_reads,
        chunk_specs=chunk_specs,
        chunk_inputs=chunk_inputs,
        chunk_outputs=chunk_outputs,
        recon_mode=recon_mode,
    )

    generated_pod5_path = pod5_dir / "generated_selected.pod5"
    generated_written = _write_generated_pod5(output_path=generated_pod5_path, reads=prepared_reads)
    if generated_written <= 0:
        raise RuntimeError("Generated validation POD5 is empty.")

    recon_per_read_path = metrics_dir / "reconstruction_per_read_metrics.jsonl"
    _write_reconstruction_per_read_metrics(recon_per_read_path, prepared_reads)

    metric_summary: dict[str, Any] = {}
    fastq_per_read: list[dict[str, Any]] = []
    real_fastq_path = fastq_dir / "original.fastq"
    generated_fastq_path = fastq_dir / "generated.fastq"
    if not args.skip_dorado:
        train_cfg = dict(config.get("train") or {})
        dorado_cfg = dict(train_cfg.get("dorado_perceptual") or {})
        dorado_model_path = args.dorado_model or dorado_cfg.get("model_path")
        dorado_model = _resolve_repo_path(dorado_model_path, config_path.parent)
        if dorado_model is None:
            raise RuntimeError(
                "No Dorado model configured. Set DORADO_MODEL or train.dorado_perceptual.model_path in the config."
            )
        dorado_bin = _resolve_dorado_bin(args.dorado_bin, cfg_dir=config_path.parent, dorado_model=dorado_model)
        _run_dorado(
            dorado_bin=str(dorado_bin),
            dorado_model=str(dorado_model),
            pod5_path=real_pod5_path,
            out_fastq=real_fastq_path,
            device=str(args.dorado_device),
        )
        _run_dorado(
            dorado_bin=str(dorado_bin),
            dorado_model=str(dorado_model),
            pod5_path=generated_pod5_path,
            out_fastq=generated_fastq_path,
            device=str(args.dorado_device),
        )
        metric_summary, fastq_per_read = _compute_metrics(real_fastq_path, generated_fastq_path)
        _write_per_read_metrics(metrics_dir / "per_read_metrics.jsonl", fastq_per_read)
        dorado_bin_text = str(dorado_bin)
        dorado_model_text = str(dorado_model)
    else:
        dorado_bin_text = None
        dorado_model_text = None

    finished_at = time.time()
    summary = {
        "status": "ok",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_split": split,
        "source_file_count": len(split_files),
        "source_files": [str(path) for path in split_files],
        "source_pod5": (None if source_pod5 is None else str(source_pod5)),
        "requested_read_count": int(num_reads),
        "selected_read_count": int(len(manifest_reads)),
        "selected_trimmed_pod5_read_count": int(real_written),
        "generated_pod5_read_count": int(generated_written),
        "min_read_length": int(min_read_length),
        "max_read_length": None if max_read_length is None else int(max_read_length),
        "segment_samples": int(segment_samples),
        "recon_mode": recon_mode,
        "recon_chunk_size": int(segment_samples),
        "recon_hop_size": int(hop_samples),
        "recon_overlap_size": int(segment_samples - hop_samples),
        "chunk_batch_size": int(chunk_batch_size),
        "trim_mode": trim_mode,
        "skip_dorado": bool(args.skip_dorado),
        "dorado_bin": dorado_bin_text,
        "dorado_model": dorado_model_text,
        "dorado_device": None if args.skip_dorado else str(args.dorado_device),
        "manifest_path": str(manifest_path),
        "started_at_unix": float(started_at),
        "finished_at_unix": float(finished_at),
        "elapsed_seconds": float(finished_at - started_at),
        "paths": {
            "original_pod5": str(real_pod5_path),
            "generated_pod5": str(generated_pod5_path),
            "reconstruction_per_read_metrics": str(recon_per_read_path),
            "original_fastq": (None if args.skip_dorado else str(real_fastq_path)),
            "generated_fastq": (None if args.skip_dorado else str(generated_fastq_path)),
            "per_read_metrics": (None if args.skip_dorado else str(metrics_dir / "per_read_metrics.jsonl")),
        },
        "selected_trimmed_lengths": trimmed_lengths,
        "warnings": manifest_warnings,
    }
    summary.update(reconstruction_summary)
    summary.update(metric_summary)
    _write_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
