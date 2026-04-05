#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pod5
import pysam

from valid.true_valid_common import TRUTH_MODE_ANALYSIS_HAC_PROXY, write_fastq_gz, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a post-training true-valid manifest from analysis outputs.")
    parser.add_argument("--analysis-root", type=Path, default=Path("/data/nanopore/hereditary_cancer_2025.09/analysis"))
    parser.add_argument("--raw-root", type=Path, default=Path("/data/nanopore/hereditary_cancer_2025.09/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for manifest and truth files.",
    )
    parser.add_argument("--fc", type=str, default="FC01")
    parser.add_argument("--barcodes", type=str, default="barcode01,barcode02,barcode03")
    parser.add_argument("--selection-mode", type=str, default="global_random")
    parser.add_argument("--quality-filter-mode", type=str, default="len_only")
    parser.add_argument("--target-total-reads", type=int, default=1000)
    parser.add_argument("--target-per-barcode", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-read-length", type=int, default=12288)
    parser.add_argument("--min-acc", type=float, default=99.0)
    parser.add_argument("--min-coverage", type=float, default=98.0)
    parser.add_argument("--min-mean-quality", type=float, default=16.0)
    parser.add_argument("--max-qstart", type=int, default=100)
    parser.add_argument("--max-tail", type=int, default=100)
    parser.add_argument("--truth-mode", type=str, default=TRUTH_MODE_ANALYSIS_HAC_PROXY)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _float_value(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    result = float(value)
    return 0.0 if math.isnan(result) else result


def _int_value(row: dict[str, str], key: str) -> int:
    return int(round(_float_value(row, key)))


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(item["analysis_acc"]),
        -float(item["analysis_coverage"]),
        -float(item["analysis_mean_quality"]),
        -int(item["analysis_read_length"]),
        str(item["read_id"]),
    )


def _stable_sample(
    *,
    items: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda item: (str(item["barcode"]), str(item["read_id"])))
    if sample_size >= len(ordered):
        return ordered
    rng = random.Random(int(seed))
    sampled_indices = sorted(rng.sample(range(len(ordered)), k=int(sample_size)))
    return [ordered[idx] for idx in sampled_indices]


def _load_candidates(
    *,
    readstats_path: Path,
    fc: str,
    barcode: str,
    min_read_length: int,
    quality_filter_mode: str,
    min_acc: float,
    min_coverage: float,
    min_mean_quality: float,
    max_qstart: int,
    max_tail: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with gzip.open(readstats_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            read_length = _int_value(row, "read_length")
            qend = _int_value(row, "qend")
            qstart = _int_value(row, "qstart")
            tail = read_length - qend
            acc = _float_value(row, "acc")
            coverage = _float_value(row, "coverage")
            mean_quality = _float_value(row, "mean_quality")
            if read_length < int(min_read_length):
                continue
            if quality_filter_mode == "strict":
                if acc < float(min_acc):
                    continue
                if coverage < float(min_coverage):
                    continue
                if mean_quality < float(min_mean_quality):
                    continue
                if qstart > int(max_qstart):
                    continue
                if tail > int(max_tail):
                    continue
            candidates.append(
                {
                    "fc": fc,
                    "barcode": barcode,
                    "read_id": str(row["name"]),
                    "analysis_ref": str(row["ref"]),
                    "analysis_direction": str(row["direction"]),
                    "analysis_rstart": _int_value(row, "rstart"),
                    "analysis_rend": _int_value(row, "rend"),
                    "analysis_qstart": qstart,
                    "analysis_qend": qend,
                    "analysis_read_length": read_length,
                    "analysis_length": _int_value(row, "length"),
                    "analysis_acc": acc,
                    "analysis_iden": _float_value(row, "iden"),
                    "analysis_coverage": coverage,
                    "analysis_ref_coverage": _float_value(row, "ref_coverage"),
                    "analysis_mean_quality": mean_quality,
                    "analysis_match": _int_value(row, "match"),
                    "analysis_ins": _int_value(row, "ins"),
                    "analysis_del": _int_value(row, "del"),
                    "analysis_sub": _int_value(row, "sub"),
                    "analysis_duplex": int(_float_value(row, "duplex")),
                    "analysis_start_time": str(row.get("start_time") or ""),
                    "analysis_runid": str(row.get("runid") or ""),
                    "analysis_sample_name": str(row.get("sample_name") or ""),
                }
            )
    return candidates


def _map_read_ids_to_pod5(
    *,
    pod5_dir: Path,
    target_ids: set[str],
) -> dict[str, dict[str, Any]]:
    remaining = set(target_ids)
    found: dict[str, dict[str, Any]] = {}
    files = sorted(pod5_dir.glob("*.pod5"))
    if not files:
        raise FileNotFoundError(f"No POD5 files found in {pod5_dir}")
    for file_path in files:
        if not remaining:
            break
        ordered = sorted(remaining)
        with pod5.Reader(str(file_path)) as reader:
            try:
                records = reader.reads(selection=ordered, missing_ok=True)
            except TypeError:
                records = reader.reads()
            for record in records:
                read_id = str(getattr(record, "read_id", ""))
                if read_id not in remaining:
                    continue
                raw_length = int(len(record.signal))
                found[read_id] = {
                    "source_file": str(file_path.resolve()),
                    "raw_length": raw_length,
                }
                remaining.discard(read_id)
    return found


def _extract_truth_from_analysis_bam(
    *,
    bam_path: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def _record_to_truth_payload(record: Any, bam: Any) -> dict[str, Any] | None:
        sequence = record.get_forward_sequence()
        try:
            qualities = record.get_forward_qualities()
        except TypeError:
            qualities = None
        if not sequence:
            return None
        quality_string = "".join(chr(int(q) + 33) for q in (qualities or []))
        if len(quality_string) != len(sequence):
            quality_string = "I" * len(sequence)
        return {
            "truth_seq": sequence,
            "truth_qual": quality_string,
            "truth_length": len(sequence),
            "analysis_hp": int(record.get_tag("HP")) if record.has_tag("HP") else None,
            "analysis_mapq": int(record.mapping_quality),
            "analysis_bam_ref": bam.get_reference_name(record.reference_id) if record.reference_id >= 0 else None,
            "analysis_bam_start": int(record.reference_start) if record.reference_start >= 0 else None,
            "analysis_bam_end": int(record.reference_end) if record.reference_end is not None else None,
            "analysis_bam_is_reverse": bool(record.is_reverse),
        }

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for item in candidates:
            read_id = str(item["read_id"])
            if read_id in found:
                continue
            ref_name = str(item["analysis_ref"])
            start = max(0, int(item["analysis_rstart"]) - 512)
            stop = max(start + 1, int(item["analysis_rend"]) + 512)
            try:
                iterator = bam.fetch(ref_name, start, stop)
            except ValueError:
                iterator = ()
            for record in iterator:
                if str(record.query_name) != read_id:
                    continue
                payload = _record_to_truth_payload(record, bam)
                if payload is None:
                    continue
                found[read_id] = payload
                break

        missing_ids = {str(item["read_id"]) for item in candidates if str(item["read_id"]) not in found}
        if missing_ids:
            for record in bam.fetch(until_eof=True):
                read_id = str(record.query_name)
                if read_id not in missing_ids:
                    continue
                payload = _record_to_truth_payload(record, bam)
                if payload is None:
                    continue
                found[read_id] = payload
                missing_ids.discard(read_id)
                if not missing_ids:
                    break
    return found


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    analysis_root = Path(args.analysis_root).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_mode = str(args.truth_mode).strip().lower()
    if truth_mode != TRUTH_MODE_ANALYSIS_HAC_PROXY:
        raise SystemExit(f"Unsupported truth mode {truth_mode!r}; currently only {TRUTH_MODE_ANALYSIS_HAC_PROXY!r} is implemented.")

    selection_mode = str(args.selection_mode).strip().lower()
    if selection_mode not in {"global_random", "per_barcode_top"}:
        raise SystemExit("Unsupported --selection-mode; choose from ['global_random', 'per_barcode_top'].")

    quality_filter_mode = str(args.quality_filter_mode).strip().lower()
    if quality_filter_mode not in {"len_only", "strict"}:
        raise SystemExit("Unsupported --quality-filter-mode; choose from ['len_only', 'strict'].")

    target_total_reads = max(0, int(args.target_total_reads))
    target_per_barcode = max(0, int(args.target_per_barcode))
    random_seed = int(args.random_seed)

    fc = str(args.fc).strip().upper()
    barcodes = [item.strip() for item in str(args.barcodes).split(",") if item.strip()]
    if not barcodes:
        raise SystemExit("At least one barcode is required.")

    candidate_rows_by_barcode: dict[str, list[dict[str, Any]]] = {}
    candidate_summary: dict[str, Any] = {}
    all_candidate_ids: set[str] = set()
    for barcode in barcodes:
        readstats_path = analysis_root / fc / barcode / f"{barcode}.readstats.tsv.gz"
        bam_path = analysis_root / fc / barcode / f"{barcode}.haplotagged.bam"
        if not readstats_path.exists():
            raise FileNotFoundError(f"Missing readstats: {readstats_path}")
        if not bam_path.exists():
            raise FileNotFoundError(f"Missing haplotagged BAM: {bam_path}")
        candidates = _load_candidates(
            readstats_path=readstats_path,
            fc=fc,
            barcode=barcode,
            min_read_length=args.min_read_length,
            quality_filter_mode=quality_filter_mode,
            min_acc=args.min_acc,
            min_coverage=args.min_coverage,
            min_mean_quality=args.min_mean_quality,
            max_qstart=args.max_qstart,
            max_tail=args.max_tail,
        )
        for item in candidates:
            item["analysis_bam"] = str(bam_path.resolve())
        candidate_rows_by_barcode[barcode] = candidates
        all_candidate_ids.update(item["read_id"] for item in candidates)
        candidate_summary[barcode] = {
            "candidate_read_count": int(len(candidates)),
        }

    pod5_map = _map_read_ids_to_pod5(
        pod5_dir=raw_root / fc / "pod5",
        target_ids=all_candidate_ids,
    )

    truth_map: dict[str, dict[str, Any]] = {}
    for barcode in barcodes:
        bam_path = analysis_root / fc / barcode / f"{barcode}.haplotagged.bam"
        barcode_candidates = list(candidate_rows_by_barcode.get(barcode, []))
        truth_map.update(_extract_truth_from_analysis_bam(bam_path=bam_path, candidates=barcode_candidates))

    usable_rows_by_barcode: dict[str, list[dict[str, Any]]] = {}
    usable_rows: list[dict[str, Any]] = []
    truth_fastq_records: list[dict[str, str]] = []
    truth_metadata_rows: list[dict[str, Any]] = []
    group_summary: dict[str, Any] = {}
    warnings: list[str] = []

    for barcode in barcodes:
        barcode_candidates = list(candidate_rows_by_barcode.get(barcode, []))
        barcode_usable: list[dict[str, Any]] = []
        for item in barcode_candidates:
            read_id = item["read_id"]
            pod5_entry = pod5_map.get(read_id)
            truth_entry = truth_map.get(read_id)
            if pod5_entry is None:
                if len(warnings) < 64:
                    warnings.append(f"Missing POD5 mapping for {barcode}:{read_id}")
                continue
            if truth_entry is None:
                if len(warnings) < 64:
                    warnings.append(f"Missing truth entry for {barcode}:{read_id}")
                continue
            merged = dict(item)
            merged.update(pod5_entry)
            merged.update(truth_entry)
            barcode_usable.append(merged)

        if selection_mode == "per_barcode_top":
            barcode_usable.sort(key=_candidate_sort_key)
        else:
            barcode_usable.sort(key=lambda item: (str(item["barcode"]), str(item["read_id"])))
        usable_rows_by_barcode[barcode] = barcode_usable
        usable_rows.extend(barcode_usable)

    if selection_mode == "global_random":
        selected_reads = _stable_sample(
            items=usable_rows,
            sample_size=target_total_reads,
            seed=random_seed,
        )
        if len(selected_reads) < target_total_reads:
            warnings.append(
                f"Requested {target_total_reads} reads in global_random mode but only found {len(selected_reads)} usable reads."
            )
    else:
        selected_reads = []
        for barcode in barcodes:
            barcode_usable = list(usable_rows_by_barcode.get(barcode, []))
            barcode_selected = barcode_usable[:target_per_barcode]
            if len(barcode_selected) < target_per_barcode:
                warnings.append(
                    f"{barcode}: requested {target_per_barcode} reads but only found {len(barcode_selected)} usable reads."
                )
            selected_reads.extend(barcode_selected)

    selected_reads.sort(key=lambda item: (str(item["barcode"]), str(item["read_id"])))
    selected_count_by_barcode = Counter(str(item["barcode"]) for item in selected_reads)

    for item in selected_reads:
        truth_fastq_records.append(
            {
                "read_id": str(item["read_id"]),
                "seq": str(item["truth_seq"]),
                "qual": str(item["truth_qual"]),
            }
        )
        truth_metadata_rows.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"truth_seq", "truth_qual"}
            }
        )

    for barcode in barcodes:
        barcode_selected = [item for item in selected_reads if str(item["barcode"]) == barcode]
        group_summary[barcode] = {
            "candidate_read_count": int(candidate_summary[barcode]["candidate_read_count"]),
            "usable_read_count": int(len(usable_rows_by_barcode.get(barcode, []))),
            "selected_read_count": int(len(barcode_selected)),
            "min_raw_length": int(min((item["raw_length"] for item in barcode_selected), default=0)),
            "max_raw_length": int(max((item["raw_length"] for item in barcode_selected), default=0)),
            "min_truth_length": int(min((item["truth_length"] for item in barcode_selected), default=0)),
            "max_truth_length": int(max((item["truth_length"] for item in barcode_selected), default=0)),
        }

    truth_dir = output_dir / "truth"
    truth_fastq_path = truth_dir / "analysis_hac_proxy_truth.fastq.gz"
    truth_metadata_path = truth_dir / "truth_metadata.jsonl"
    manifest_path = output_dir / "manifest" / f"true_valid_{fc.lower()}_{len(selected_reads)}reads_manifest.json"

    manifest_payload = {
        "status": "ok",
        "truth_mode": truth_mode,
        "analysis_root": str(analysis_root),
        "raw_root": str(raw_root),
        "fc": fc,
        "barcodes": barcodes,
        "selection": {
            "selection_mode": selection_mode,
            "quality_filter_mode": quality_filter_mode,
            "random_seed": int(random_seed),
            "target_total_reads": int(target_total_reads),
            "target_per_barcode": int(target_per_barcode),
            "min_read_length": int(args.min_read_length),
            "min_acc": float(args.min_acc),
            "min_coverage": float(args.min_coverage),
            "min_mean_quality": float(args.min_mean_quality),
            "max_qstart": int(args.max_qstart),
            "max_tail": int(args.max_tail),
            "applied_filters": ["min_read_length"] if quality_filter_mode == "len_only" else [
                "min_read_length",
                "min_acc",
                "min_coverage",
                "min_mean_quality",
                "max_qstart",
                "max_tail",
            ],
            "sort_order": (
                ["analysis_acc", "analysis_coverage", "analysis_mean_quality", "analysis_read_length"]
                if selection_mode == "per_barcode_top"
                else ["barcode", "read_id"]
            ),
        },
        "candidate_read_count": int(sum(len(items) for items in candidate_rows_by_barcode.values())),
        "usable_read_count": int(len(usable_rows)),
        "selected_read_count": int(len(selected_reads)),
        "selected_barcode_counts": {barcode: int(selected_count_by_barcode.get(barcode, 0)) for barcode in barcodes},
        "group_summary": group_summary,
        "selected_reads": [
            {key: value for key, value in item.items() if key not in {"truth_seq", "truth_qual"}}
            for item in selected_reads
        ],
        "paths": {
            "truth_fastq": str(truth_fastq_path),
            "truth_metadata": str(truth_metadata_path),
        },
        "warnings": warnings,
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
    }
    if args.dry_run:
        dry_run_payload = dict(manifest_payload)
        dry_run_payload["selected_read_examples"] = dry_run_payload.get("selected_reads", [])[:10]
        dry_run_payload.pop("selected_reads", None)
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return

    write_fastq_gz(truth_fastq_path, truth_fastq_records)
    write_jsonl(truth_metadata_path, truth_metadata_rows)
    write_json(manifest_path, manifest_payload)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "candidate_read_count": manifest_payload["candidate_read_count"],
                "usable_read_count": manifest_payload["usable_read_count"],
                "selected_read_count": len(selected_reads),
                "selected_barcode_counts": manifest_payload["selected_barcode_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
