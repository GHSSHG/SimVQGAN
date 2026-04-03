#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pod5
from pod5 import Writer

from valid.export_valid_recon_pod5 import _load_json, _resolve_segment_hop_samples, _resolve_segment_samples
from valid.run_validation import (
    RECON_MODE_DIRECT,
    RECON_MODE_OVERLAP,
    ChunkSpec,
    ReconstructionRead,
    _CheckpointReconstructor,
    _count_reads,
    _compute_metrics,
    _fetch_records_by_id,
    _finalize_reconstruction,
    _progress_markers,
    _resolve_dorado_bin,
    _resolve_recon_hop,
    _resolve_repo_path,
    _run_dorado,
    _summarize,
    _write_generated_pod5,
    _write_per_read_metrics,
    _write_reconstruction_per_read_metrics,
)
from codec.data.normalization import normalize_to_pm1_with_stats
from codec.data.pod5_processing import parse_calibration
from valid.true_valid_common import (
    compare_fastq_to_truth,
    compute_summary_delta,
    group_manifest_reads,
    load_truth_entries,
    read_json,
    selected_manifest_reads_to_base_manifest,
    summarize_group_metric_records,
    truth_mode_text,
    write_json,
)


@dataclass(frozen=True)
class ShiftLastWindow:
    start: int
    stop: int
    valid_from: int
    valid_to: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training true-valid for one checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-batch-size", type=int, default=128)
    parser.add_argument("--recon-mode", type=str, default=RECON_MODE_OVERLAP)
    parser.add_argument("--hop-samples", type=int, default=None)
    parser.add_argument("--trim-mode", type=str, default="drop")
    parser.add_argument("--tail-chunk-mode", type=str, default="shift_last")
    parser.add_argument("--dorado-bin", type=str, default="dorado")
    parser.add_argument("--dorado-model", type=str, default=None)
    parser.add_argument("--dorado-device", type=str, default="cuda:0")
    return parser.parse_args()


def _build_shift_last_windows(total_samples: int, chunk_size: int, hop_size: int) -> list[ShiftLastWindow]:
    n = int(total_samples)
    chunk_size = int(chunk_size)
    hop_size = int(hop_size)
    if n < chunk_size:
        return []
    last_start = n - chunk_size
    starts = list(range(0, last_start + 1, hop_size))
    if not starts:
        starts = [0]
    windows = [
        ShiftLastWindow(start=int(start), stop=int(start + chunk_size), valid_from=0, valid_to=chunk_size)
        for start in starts
    ]
    if starts[-1] != last_start:
        prev_start = starts[-1]
        valid_global_start = prev_start + hop_size
        local_valid_from = max(0, int(valid_global_start - last_start))
        windows.append(
            ShiftLastWindow(
                start=int(last_start),
                stop=int(last_start + chunk_size),
                valid_from=local_valid_from,
                valid_to=chunk_size,
            )
        )
    return windows


def _shift_last_window_weights(
    *,
    window_index: int,
    window_count: int,
    valid_from: int,
    valid_to: int,
    chunk_size: int,
    hop_size: int,
) -> np.ndarray:
    weights = np.zeros((int(chunk_size),), dtype=np.float32)
    valid_from = max(0, int(valid_from))
    valid_to = min(int(chunk_size), int(valid_to))
    if valid_to <= valid_from:
        return weights
    weights[valid_from:valid_to] = 1.0
    overlap_size = int(chunk_size) - int(hop_size)
    active_len = valid_to - valid_from
    if overlap_size <= 0 or window_count <= 1 or active_len <= 0:
        return weights
    fade_len = min(overlap_size, active_len)
    if window_index > 0 and fade_len > 0:
        fade_in = np.arange(fade_len, dtype=np.float32) / float(fade_len)
        weights[valid_from : valid_from + fade_len] = fade_in
    if (window_index + 1) < window_count and fade_len > 0:
        fade_out = 1.0 - (np.arange(fade_len, dtype=np.float32) / float(fade_len))
        weights[valid_to - fade_len : valid_to] = np.minimum(weights[valid_to - fade_len : valid_to], fade_out)
    return weights


def _write_selected_real_pod5_shift_last(*, manifest_reads: list[Any], output_path: Path) -> tuple[int, dict[str, int]]:
    grouped: dict[str, list[Any]] = {}
    for item in manifest_reads:
        grouped.setdefault(item.source_file, []).append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    expected = len(manifest_reads)
    written = 0
    read_lengths: dict[str, int] = {}
    markers = _progress_markers(expected)
    print(f"[real] writing selected POD5 -> {output_path}", flush=True)
    with Writer(str(output_path)) as writer:
        for file_path_str, items in grouped.items():
            ordered_ids = [item.read_id for item in items]
            with pod5.Reader(str(Path(file_path_str))) as reader:
                fetched = _fetch_records_by_id(reader, ordered_ids)
                for item in items:
                    record = fetched[item.read_id]
                    signal = np.asarray(record.signal, dtype=np.int16)
                    read = record.to_read()
                    read.signal = signal
                    writer.add_read(read)
                    read_lengths[item.read_id] = int(signal.size)
                    written += 1
                    if markers and written >= markers[0]:
                        print(f"[real] progress {written}/{expected}", flush=True)
                        markers.pop(0)
    print(f"[real] done. kept {written}/{expected} reads", flush=True)
    return written, read_lengths


def _load_prepared_reads_shift_last(
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
            raw = np.asarray(record.signal, dtype=np.int16).reshape(-1)
            calibration = parse_calibration(getattr(record, "calibration", None))
            signal_pa = calibration.to_picoamps(raw)
            windows = _build_shift_last_windows(int(raw.size), segment_samples, hop_samples)
            if not windows:
                raise RuntimeError(f"Selected POD5 read has no reconstructable chunks: {read_id}")
            overlap_weights = None
            if recon_mode == RECON_MODE_OVERLAP:
                overlap_weights = [
                    _shift_last_window_weights(
                        window_index=window_index,
                        window_count=len(windows),
                        valid_from=window.valid_from,
                        valid_to=window.valid_to,
                        chunk_size=segment_samples,
                        hop_size=hop_samples,
                    )
                    for window_index, window in enumerate(windows)
                ]
            state = ReconstructionRead(
                read_index=read_index,
                source_file=str(source_pod5),
                read_id=read_id,
                raw_length=int(raw.size),
                trimmed_length=int(raw.size),
                chunk_count=len(windows),
                calibration=calibration,
                template_read=record.to_read(),
                trimmed_raw=raw,
                trimmed_pa=np.asarray(signal_pa, dtype=np.float32),
                chunk_starts=[int(window.start) for window in windows],
                overlap_weights=overlap_weights,
            )
            if recon_mode == RECON_MODE_DIRECT:
                state.reconstructed_pa = np.zeros((int(raw.size),), dtype=np.float32)
                state.reconstructed_adc = np.zeros((int(raw.size),), dtype=np.int16)
            else:
                state.pa_acc = np.zeros((int(raw.size),), dtype=np.float32)
                state.weight_acc = np.zeros((int(raw.size),), dtype=np.float32)
            reads.append(state)
            for chunk_index, window in enumerate(windows):
                chunk_pa = np.asarray(signal_pa[window.start : window.stop], dtype=np.float32)
                normalized, center, half_range = normalize_to_pm1_with_stats(chunk_pa)
                chunks.append(np.asarray(normalized, dtype=np.float32))
                chunk_specs.append(
                    ChunkSpec(
                        read_index=read_index,
                        read_id=read_id,
                        chunk_index=chunk_index,
                        start=int(window.start),
                        stop=int(window.stop),
                        center=float(center),
                        half_range=float(half_range),
                    )
                )
    if not chunks:
        raise RuntimeError("Prepared chunk pool is empty.")
    print(f"[prep] done. reads={len(reads)} chunks={len(chunks)}", flush=True)
    return reads, chunk_specs, np.stack(chunks, axis=0).astype(np.float32)


def _compute_group_summaries(
    *,
    manifest_payload: dict[str, Any],
    original_records: list[dict[str, Any]],
    generated_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_manifest_reads(manifest_payload, "barcode")
    summary: dict[str, Any] = {}
    for barcode, items in grouped.items():
        read_ids = [str(item["read_id"]) for item in items]
        original_summary = summarize_group_metric_records(metric_records=original_records, grouped_read_ids=read_ids)
        generated_summary = summarize_group_metric_records(metric_records=generated_records, grouped_read_ids=read_ids)
        summary[barcode] = {
            "selected_manifest_read_count": int(len(read_ids)),
            "original_vs_truth": original_summary,
            "generated_vs_truth": generated_summary,
            "generated_minus_original": compute_summary_delta(generated_summary, original_summary),
        }
    return summary


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(config_path)
    config["_config_dir"] = str(config_path.parent.resolve())
    manifest_payload = read_json(manifest_path)
    truth_entries = load_truth_entries(manifest_payload)
    split_name = str(manifest_payload.get("fc") or "FC01").strip().lower()
    split_name = "valid_" + split_name if not split_name.startswith("valid_") else split_name
    data_cfg = dict(config.get("data") or {})
    if split_name in data_cfg:
        resolved_split_name = split_name
    elif "valid_fc01" in data_cfg:
        resolved_split_name = "valid_fc01"
    elif "valid" in data_cfg:
        resolved_split_name = "valid"
    else:
        raise RuntimeError(
            f"Unable to resolve true-valid split for manifest fc={manifest_payload.get('fc')!r}; "
            f"looked for {split_name!r}, 'valid_fc01', and 'valid'."
        )
    segment_samples = _resolve_segment_samples(config, resolved_split_name)
    configured_hop_samples = _resolve_segment_hop_samples(
        config,
        resolved_split_name,
    )
    hop_samples = _resolve_recon_hop(args.recon_mode, segment_samples, configured_hop_samples, args.hop_samples)

    pod5_dir = output_dir / "pod5"
    fastq_dir = output_dir / "fastq"
    metrics_dir = output_dir / "metrics"
    pod5_dir.mkdir(parents=True, exist_ok=True)
    fastq_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_reads = selected_manifest_reads_to_base_manifest(manifest_payload)
    original_pod5_path = pod5_dir / "original_selected.pod5"
    tail_chunk_mode = str(args.tail_chunk_mode).strip().lower() or "shift_last"
    if tail_chunk_mode not in {"drop", "shift_last"}:
        raise RuntimeError(f"Unsupported tail chunk mode: {tail_chunk_mode}")
    if tail_chunk_mode == "shift_last":
        original_written, trimmed_lengths = _write_selected_real_pod5_shift_last(
            manifest_reads=manifest_reads,
            output_path=original_pod5_path,
        )
    else:
        from valid.run_validation import _write_selected_real_pod5

        original_written, trimmed_lengths = _write_selected_real_pod5(
            manifest_reads=manifest_reads,
            output_path=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            trim_mode=args.trim_mode,
        )
    if original_written <= 0:
        raise RuntimeError("True-valid original POD5 is empty.")

    if tail_chunk_mode == "shift_last":
        prepared_reads, chunk_specs, chunk_inputs = _load_prepared_reads_shift_last(
            source_pod5=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            recon_mode=args.recon_mode,
        )
    else:
        from valid.run_validation import _load_prepared_reads

        prepared_reads, chunk_specs, chunk_inputs = _load_prepared_reads(
            source_pod5=original_pod5_path,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            recon_mode=args.recon_mode,
        )
    reconstructor = _CheckpointReconstructor(
        model_cfg=dict(config.get("model") or {}),
        checkpoint_path=checkpoint_path,
        chunk_batch_size=max(1, int(args.chunk_batch_size)),
    )
    chunk_outputs = reconstructor.reconstruct_chunks(chunk_inputs)
    reconstruction_summary = _finalize_reconstruction(
        reads=prepared_reads,
        chunk_specs=chunk_specs,
        chunk_inputs=chunk_inputs,
        chunk_outputs=chunk_outputs,
        recon_mode=args.recon_mode,
    )

    generated_pod5_path = pod5_dir / "generated_selected.pod5"
    generated_written = _write_generated_pod5(output_path=generated_pod5_path, reads=prepared_reads)
    if generated_written <= 0:
        raise RuntimeError("True-valid generated POD5 is empty.")

    recon_metrics_path = metrics_dir / "reconstruction_per_read_metrics.jsonl"
    _write_reconstruction_per_read_metrics(recon_metrics_path, prepared_reads)

    train_cfg = dict(config.get("train") or {})
    dorado_cfg = dict(train_cfg.get("dorado_perceptual") or {})
    dorado_model_path = args.dorado_model or dorado_cfg.get("model_path")
    dorado_model = _resolve_repo_path(dorado_model_path, config_path.parent)
    if dorado_model is None:
        raise RuntimeError("No Dorado model configured for true valid.")
    dorado_bin = _resolve_dorado_bin(args.dorado_bin, cfg_dir=config_path.parent, dorado_model=dorado_model)

    original_fastq_path = fastq_dir / "original.fastq"
    generated_fastq_path = fastq_dir / "generated.fastq"
    _run_dorado(
        dorado_bin=str(dorado_bin),
        dorado_model=str(dorado_model),
        pod5_path=original_pod5_path,
        out_fastq=original_fastq_path,
        device=args.dorado_device,
    )
    _run_dorado(
        dorado_bin=str(dorado_bin),
        dorado_model=str(dorado_model),
        pod5_path=generated_pod5_path,
        out_fastq=generated_fastq_path,
        device=args.dorado_device,
    )

    original_vs_truth_summary, original_vs_truth_records = compare_fastq_to_truth(
        predicted_fastq=original_fastq_path,
        truth_entries=truth_entries,
    )
    generated_vs_truth_summary, generated_vs_truth_records = compare_fastq_to_truth(
        predicted_fastq=generated_fastq_path,
        truth_entries=truth_entries,
    )
    original_vs_generated_summary, original_vs_generated_records = _compute_metrics(original_fastq_path, generated_fastq_path)

    _write_per_read_metrics(metrics_dir / "original_vs_truth.jsonl", original_vs_truth_records)
    _write_per_read_metrics(metrics_dir / "generated_vs_truth.jsonl", generated_vs_truth_records)
    _write_per_read_metrics(metrics_dir / "original_vs_generated.jsonl", original_vs_generated_records)

    per_barcode_summary = _compute_group_summaries(
        manifest_payload=manifest_payload,
        original_records=original_vs_truth_records,
        generated_records=generated_vs_truth_records,
    )
    write_json(metrics_dir / "per_barcode_summary.json", per_barcode_summary)

    summary = {
        "status": "ok",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "truth_mode": truth_mode_text(manifest_payload),
        "selected_read_count": int(len(manifest_payload.get("selected_reads", []))),
        "selected_trimmed_pod5_read_count": int(original_written),
        "generated_pod5_read_count": int(generated_written),
        "segment_samples": int(segment_samples),
        "recon_mode": str(args.recon_mode),
        "recon_hop_size": int(hop_samples),
        "trim_mode": str(args.trim_mode),
        "tail_chunk_mode": tail_chunk_mode,
        "chunk_batch_size": int(args.chunk_batch_size),
        "dorado_bin": str(dorado_bin),
        "dorado_model": str(dorado_model),
        "dorado_device": str(args.dorado_device),
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
        "elapsed_seconds": float(time.time() - started_at),
        "paths": {
            "original_pod5": str(original_pod5_path),
            "generated_pod5": str(generated_pod5_path),
            "original_fastq": str(original_fastq_path),
            "generated_fastq": str(generated_fastq_path),
            "reconstruction_per_read_metrics": str(recon_metrics_path),
            "original_vs_truth": str(metrics_dir / "original_vs_truth.jsonl"),
            "generated_vs_truth": str(metrics_dir / "generated_vs_truth.jsonl"),
            "original_vs_generated": str(metrics_dir / "original_vs_generated.jsonl"),
            "per_barcode_summary": str(metrics_dir / "per_barcode_summary.json"),
        },
        "selected_trimmed_lengths": trimmed_lengths,
        "original_vs_truth": original_vs_truth_summary,
        "generated_vs_truth": generated_vs_truth_summary,
        "generated_minus_original_vs_truth": compute_summary_delta(generated_vs_truth_summary, original_vs_truth_summary),
        "original_vs_generated": original_vs_generated_summary,
        "warnings": manifest_payload.get("warnings", []),
    }
    summary.update(reconstruction_summary)
    write_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
