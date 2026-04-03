from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from valid.run_validation import _alignment_identity, _mean_qscore, _read_fastq, _summarize

TRUTH_MODE_ANALYSIS_HAC_PROXY = "analysis_hac_proxy"
EMPIRICAL_QSCORE_CAP = 60.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_fastq_gz(path: Path, records: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"@{record['read_id']}\n")
            handle.write(f"{record['seq']}\n+\n")
            handle.write(f"{record['qual']}\n")


def load_truth_entries(manifest_payload: dict[str, Any]) -> dict[str, Any]:
    truth_path = Path(manifest_payload["paths"]["truth_fastq"]).expanduser().resolve()
    return _read_fastq(truth_path)


def empirical_qscore(edit_distance: int, denom: int, *, cap: float = EMPIRICAL_QSCORE_CAP) -> float:
    denom = max(1, int(denom))
    errors = max(0, int(edit_distance))
    if errors <= 0:
        return float(cap)
    rate = max(float(errors) / float(denom), 1e-12)
    return float(min(cap, -10.0 * math.log10(rate)))


def _diff_summary(new_value: float, old_value: float) -> float:
    return float(new_value - old_value)


def metric_summary_from_records(
    records: Sequence[dict[str, Any]],
    *,
    truth_read_count: int,
    predicted_read_count: int,
) -> dict[str, Any]:
    identities = [float(record["identity"]) for record in records]
    empirical_qscores = [float(record["empirical_qscore"]) for record in records]
    predicted_qscores = [float(record["predicted_qscore"]) for record in records]
    truth_proxy_qscores = [float(record["truth_proxy_qscore"]) for record in records]
    length_deltas = [float(record["length_delta"]) for record in records]
    total_weight = int(sum(int(record["identity_weight"]) for record in records))
    total_identity_weight = float(
        sum(float(record["identity"]) * float(record["identity_weight"]) for record in records)
    )
    exact_matches = int(sum(1 for record in records if bool(record["exact_match"])))
    shared_count = int(len(records))
    return {
        "shared_read_count": shared_count,
        "truth_read_count": int(truth_read_count),
        "predicted_read_count": int(predicted_read_count),
        "truth_only_read_count": int(max(0, truth_read_count - shared_count)),
        "predicted_only_read_count": int(max(0, predicted_read_count - shared_count)),
        "exact_match_rate": (float(exact_matches) / float(shared_count)) if shared_count else 0.0,
        "length_weighted_identity": (float(total_identity_weight) / float(total_weight)) if total_weight else 0.0,
        "mean_predicted_qscore": float(np.mean(np.asarray(predicted_qscores, dtype=np.float64))) if predicted_qscores else 0.0,
        "mean_truth_proxy_qscore": (
            float(np.mean(np.asarray(truth_proxy_qscores, dtype=np.float64))) if truth_proxy_qscores else 0.0
        ),
        "mean_empirical_qscore": (
            float(np.mean(np.asarray(empirical_qscores, dtype=np.float64))) if empirical_qscores else 0.0
        ),
        "identity_summary": _summarize(identities),
        "empirical_qscore_summary": _summarize(empirical_qscores),
        "predicted_qscore_summary": _summarize(predicted_qscores),
        "truth_proxy_qscore_summary": _summarize(truth_proxy_qscores),
        "length_delta_summary": _summarize(length_deltas),
    }


def compare_fastq_to_truth(
    *,
    predicted_fastq: Path,
    truth_entries: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_entries = _read_fastq(predicted_fastq)
    shared_ids = sorted(set(predicted_entries) & set(truth_entries))
    records: list[dict[str, Any]] = []
    for read_id in shared_ids:
        truth_entry = truth_entries[read_id]
        predicted_entry = predicted_entries[read_id]
        identity, edit_distance, denom, backend = _alignment_identity(truth_entry.seq, predicted_entry.seq)
        record = {
            "read_id": read_id,
            "truth_length": int(len(truth_entry.seq)),
            "predicted_length": int(len(predicted_entry.seq)),
            "identity": float(identity),
            "edit_distance": int(edit_distance),
            "identity_weight": int(denom),
            "predicted_qscore": float(_mean_qscore(predicted_entry.qual)),
            "truth_proxy_qscore": float(_mean_qscore(truth_entry.qual)),
            "empirical_qscore": float(empirical_qscore(edit_distance, denom)),
            "length_delta": float(len(predicted_entry.seq) - len(truth_entry.seq)),
            "exact_match": bool(truth_entry.seq == predicted_entry.seq),
            "alignment_backend": backend,
        }
        records.append(record)
    summary = metric_summary_from_records(
        records,
        truth_read_count=len(truth_entries),
        predicted_read_count=len(predicted_entries),
    )
    summary["alignment_backend"] = records[0]["alignment_backend"] if records else "none"
    return summary, records


def summarize_group_metric_records(
    *,
    metric_records: Sequence[dict[str, Any]],
    grouped_read_ids: Sequence[str],
) -> dict[str, Any]:
    read_id_set = set(grouped_read_ids)
    selected = [record for record in metric_records if record["read_id"] in read_id_set]
    return metric_summary_from_records(
        selected,
        truth_read_count=len(read_id_set),
        predicted_read_count=len(selected),
    )


def compute_summary_delta(new_summary: dict[str, Any], old_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_shared_read_count": int(new_summary.get("shared_read_count", 0) - old_summary.get("shared_read_count", 0)),
        "delta_length_weighted_identity": _diff_summary(
            float(new_summary.get("length_weighted_identity", 0.0)),
            float(old_summary.get("length_weighted_identity", 0.0)),
        ),
        "delta_mean_empirical_qscore": _diff_summary(
            float(new_summary.get("mean_empirical_qscore", 0.0)),
            float(old_summary.get("mean_empirical_qscore", 0.0)),
        ),
        "delta_mean_predicted_qscore": _diff_summary(
            float(new_summary.get("mean_predicted_qscore", 0.0)),
            float(old_summary.get("mean_predicted_qscore", 0.0)),
        ),
        "delta_exact_match_rate": _diff_summary(
            float(new_summary.get("exact_match_rate", 0.0)),
            float(old_summary.get("exact_match_rate", 0.0)),
        ),
    }


def group_manifest_reads(manifest_payload: dict[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in manifest_payload.get("selected_reads", []):
        group_value = str(item.get(key) or "unknown")
        grouped.setdefault(group_value, []).append(item)
    return grouped


def selected_manifest_reads_to_base_manifest(manifest_payload: dict[str, Any]) -> list[Any]:
    from valid.run_validation import ManifestRead

    return [
        ManifestRead(
            source_file=str(item["source_file"]),
            read_id=str(item["read_id"]),
            raw_length=int(item["raw_length"]),
        )
        for item in manifest_payload.get("selected_reads", [])
    ]


def truth_mode_text(manifest_payload: dict[str, Any]) -> str:
    return str(manifest_payload.get("truth_mode") or TRUTH_MODE_ANALYSIS_HAC_PROXY)

