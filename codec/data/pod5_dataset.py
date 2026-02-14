from __future__ import annotations

import queue
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import pod5 as p5

from .pod5_processing import (
    normalize_adc_signal,
    resolve_sample_rate,
    NormalizationStats,
    CalibrationParams,
    CalibrationError,
)


def _normalize_read_signal(signal: np.ndarray, calibration: Any) -> tuple[np.ndarray, NormalizationStats, CalibrationParams]:
    """Convert int16 ADC to normalized float32 using POD5 calibration."""
    normalized, stats, cal = normalize_adc_signal(signal, calibration)
    return normalized, stats, cal


def _iter_full_chunks(signal: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    n = int(signal.shape[0])
    if n <= 0 or chunk_size <= 0:
        return
    num_full = n // chunk_size
    if num_full <= 0:
        return
    view = signal[: num_full * chunk_size]
    view = view.reshape(num_full, chunk_size)
    for row in view:
        yield row


def _should_skip_pod5(exc: Exception) -> bool:
    msg = str(exc)
    keywords = (
        "Invalid signature in file",
        "Failed to open pod5 file",
        "Pod5ApiException",
        "Bad address",
        "Error writing bytes to file",
    )
    return any(k in msg for k in keywords)


"""POD5 dataset utilities.

Each read is streamed from POD5, converted to picoamps via calibration, scaled
once to [-1, 1] using its own min/max, and then chunked so every window shares
the same per-read statistics. Sample-rate hints stored in POD5 are preferred,
falling back to configured defaults only when metadata is absent.
"""


@dataclass
class NanoporeSignalDataset:
    pod5_files: List[Path]
    window_ms: int = 1000
    window_samples: Optional[int] = None
    sample_rate_hz_default: Optional[float] = None
    return_metadata: bool = False
    read_ids_per_file: Optional[Dict[Path, Sequence[str]]] = None
    loader_workers: int = 1
    loader_prefetch_chunks: int = 128
    _invalid_files: set[Path] = field(default_factory=set, init=False, repr=False)
    _cached_length: Optional[int] = field(default=None, init=False, repr=False)
    _calibration_warned_files: set[Path] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_paths(
        cls,
        files: Iterable[Union[str, Path]],
        window_ms: int = 1000,
        window_samples: Optional[int] = None,
        sample_rate_hz_default: Optional[float] = None,
        return_metadata: bool = False,
        read_ids_per_file: Optional[Dict[Union[str, Path], Sequence[str]]] = None,
        loader_workers: int = 1,
        loader_prefetch_chunks: int = 128,
    ) -> "NanoporeSignalDataset":
        paths = [Path(f).expanduser().resolve() for f in files]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"POD5 not found: {p}")
        rid_map: Optional[Dict[Path, Sequence[str]]] = None
        if read_ids_per_file:
            rid_map = {Path(k).expanduser().resolve(): v for k, v in read_ids_per_file.items()}
        workers = max(1, int(loader_workers))
        prefetch_chunks = max(1, int(loader_prefetch_chunks))
        window_samples_int: Optional[int] = None
        if window_samples is not None:
            ws = int(window_samples)
            if ws > 0:
                window_samples_int = ws
        return cls(
            pod5_files=paths,
            window_ms=int(window_ms),
            window_samples=window_samples_int,
            sample_rate_hz_default=sample_rate_hz_default,
            return_metadata=bool(return_metadata),
            read_ids_per_file=rid_map,
            loader_workers=workers,
            loader_prefetch_chunks=prefetch_chunks,
        )

    def _iter_chunks_from_file(self, file_path: Path) -> Iterator[np.ndarray]:
        if file_path in self._invalid_files:
            return
        selection = None
        if self.read_ids_per_file and file_path in self.read_ids_per_file:
            selection = self.read_ids_per_file[file_path]

        def _mark_bad_file(exc: Exception, context: str) -> None:
            if file_path in self._invalid_files:
                return
            print(f"[warn] {context}，永久跳过 {file_path}: {exc}")
            self._invalid_files.add(file_path)

        warned_sr_mismatch = False
        try:
            with p5.Reader(str(file_path)) as reader:
                run_info = getattr(reader, "run_info", None)
                gen = reader.reads(selection=selection) if selection else reader.reads()
                for read in gen:
                    try:
                        read_sr = getattr(read, "sample_rate", None)
                        run_sr = getattr(run_info, "sample_rate", None)
                        measured_sr = None
                        if read_sr is not None:
                            try:
                                measured_sr = float(read_sr)
                            except Exception:
                                measured_sr = None
                        if measured_sr is None and run_sr is not None:
                            try:
                                measured_sr = float(run_sr)
                            except Exception:
                                measured_sr = None

                        target_sr = resolve_sample_rate(
                            read_obj=read,
                            run_info=run_info,
                            configured_hz=self.sample_rate_hz_default,
                        )
                        # Only gate when we actually have a measured value from metadata.
                        if measured_sr is not None and self.sample_rate_hz_default is not None:
                            mismatch = abs(measured_sr - float(self.sample_rate_hz_default))
                            tol = max(1.0, 0.001 * float(self.sample_rate_hz_default))
                            if mismatch > tol:
                                if not warned_sr_mismatch:
                                    print(
                                        f"[warn] read sample_rate={measured_sr:.2f}Hz differs from configured {float(self.sample_rate_hz_default):.2f}Hz; skipping reads in {file_path}",
                                        flush=True,
                                    )
                                    warned_sr_mismatch = True
                                continue
                        if self.window_samples is not None and self.window_samples > 0:
                            chunk_size = int(self.window_samples)
                        else:
                            chunk_size = int(round(self.window_ms * float(target_sr) / 1000.0))
                        if chunk_size <= 0:
                            chunk_size = 1
                        raw_signal = read.signal
                        if raw_signal.shape[0] < chunk_size:
                            continue
                        try:
                            # Normalize the full read prior to slicing windows to keep stats consistent across chunks.
                            norm_signal, stats, cal = _normalize_read_signal(
                                raw_signal, getattr(read, "calibration", None)
                            )
                        except CalibrationError as cal_exc:
                            if file_path not in self._calibration_warned_files:
                                read_id = getattr(read, "read_id", "unknown")
                                warnings.warn(
                                    f"[pod5] Skipping reads in {file_path.name} (first failing read {read_id}): {cal_exc}",
                                    RuntimeWarning,
                                )
                                self._calibration_warned_files.add(file_path)
                            continue
                        for chunk in _iter_full_chunks(norm_signal, chunk_size=chunk_size):
                            arr = np.asarray(chunk, dtype=np.float32)
                            if not self.return_metadata:
                                yield arr
                            else:
                                yield (arr, stats, cal)
                    except Exception as read_exc:
                        if _should_skip_pod5(read_exc):
                            _mark_bad_file(read_exc, "读取 read 失败")
                            return
                        read_id = getattr(read, "read_id", "unknown")
                        raise RuntimeError(f"读取 {file_path} 中 read {read_id} 失败: {read_exc}") from read_exc
        except Exception as open_exc:
            if _should_skip_pod5(open_exc):
                _mark_bad_file(open_exc, "POD5 文件损坏")
                return
            raise RuntimeError(f"打开 POD5 {file_path} 失败: {open_exc}") from open_exc

    @staticmethod
    def _flush_batch(buf: List[np.ndarray]) -> np.ndarray:
        if buf and isinstance(buf[0], tuple):
            raise ValueError("return_metadata=True 数据集不支持批量堆叠；请迭代 chunk 自行处理元数据")
        batch = np.asarray(np.stack(buf, axis=0), dtype=np.float32)
        return batch[:, np.newaxis, :]

    def iter_chunks(self, files_cycle: bool = False) -> Iterator[np.ndarray]:
        while True:
            for fp in self.pod5_files:
                if fp in self._invalid_files:
                    continue
                yield from self._iter_chunks_from_file(fp)
            if not files_cycle:
                break

    def batches(
        self,
        batch_size: int,
        drop_last: bool = True,
        files_cycle: bool = False,
        num_workers: Optional[int] = None,
        max_chunk_queue: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.return_metadata:
            raise ValueError("return_metadata=True 时请使用 iter_chunks 手动消费数据")
        worker_count = int(num_workers if num_workers is not None else self.loader_workers)
        queue_cap = int(max_chunk_queue if max_chunk_queue is not None else self.loader_prefetch_chunks)
        # Threaded path engages whenever >1 worker is requested; finite epochs still benefit
        if worker_count <= 1:
            while True:
                buf: List[np.ndarray] = []
                for fp in self.pod5_files:
                    if fp in self._invalid_files:
                        continue
                    for chunk in self._iter_chunks_from_file(fp):
                        buf.append(chunk)
                        if len(buf) == batch_size:
                            yield self._flush_batch(buf)
                            buf.clear()
                if not drop_last and len(buf) > 0:
                    yield self._flush_batch(buf)
                    buf.clear()
                if not files_cycle:
                    break
            return
        yield from self._threaded_batches(
            batch_size=batch_size,
            drop_last=drop_last,
            files_cycle=files_cycle,
            worker_count=worker_count,
            max_chunk_queue=max(1, queue_cap),
        )

    class _FileIterator:
        def __init__(self, files: List[Path], files_cycle: bool):
            self._files = files
            self._files_cycle = files_cycle
            self._idx = 0
            self._lock = threading.Lock()

        def next(self) -> Optional[Path]:
            with self._lock:
                if not self._files:
                    return None
                if self._idx >= len(self._files):
                    if not self._files_cycle:
                        return None
                    self._idx = 0
                fp = self._files[self._idx]
                self._idx += 1
                return fp

    def _threaded_batches(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        files_cycle: bool,
        worker_count: int,
        max_chunk_queue: int,
    ) -> Iterator[np.ndarray]:
        valid_files = [fp for fp in self.pod5_files if fp not in self._invalid_files]
        if not valid_files:
            raise FileNotFoundError("No valid POD5 files to stream from.")
        sentinel = object()
        chunk_queue: "queue.Queue[object]" = queue.Queue(maxsize=max_chunk_queue)
        stop_event = threading.Event()
        file_iter = self._FileIterator(valid_files, files_cycle)
        workers: list[threading.Thread] = []

        def worker_main() -> None:
            try:
                while not stop_event.is_set():
                    fp = file_iter.next()
                    if fp is None:
                        break
                    for chunk in self._iter_chunks_from_file(fp):
                        if stop_event.is_set():
                            break
                        chunk_queue.put(chunk, block=True)
            except Exception as exc:
                chunk_queue.put(exc)
            finally:
                chunk_queue.put(sentinel)

        for _ in range(worker_count):
            t = threading.Thread(target=worker_main, daemon=True)
            t.start()
            workers.append(t)

        def generator() -> Iterator[np.ndarray]:
            active = worker_count
            buf: List[np.ndarray] = []
            try:
                while True:
                    try:
                        item = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        if stop_event.is_set() and (not files_cycle or active == 0):
                            break
                        continue
                    if item is sentinel:
                        active -= 1
                        # When streaming forever we only break once stop_event is set (consumer closed) or workers drained
                        if active == 0 and (not files_cycle or stop_event.is_set()):
                            break
                        continue
                    if isinstance(item, Exception):
                        raise item
                    buf.append(item)  # type: ignore[arg-type]
                    if len(buf) == batch_size:
                        yield self._flush_batch(buf)
                        buf.clear()
            finally:
                stop_event.set()
                # Drain remaining sentinels so worker threads can exit
                while active > 0:
                    try:
                        item = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is sentinel:
                        active -= 1
                for t in workers:
                    t.join(timeout=0.5)
            if not drop_last and buf:
                yield self._flush_batch(buf)

        yield from generator()

    def __len__(self) -> int:
        if self._cached_length is not None:
            return self._cached_length
        if self.read_ids_per_file:
            total = sum(len(ids) for ids in self.read_ids_per_file.values())
        else:
            valid = [fp for fp in self.pod5_files if fp not in self._invalid_files]
            total = len(valid)
        self._cached_length = max(0, total)
        return self._cached_length
