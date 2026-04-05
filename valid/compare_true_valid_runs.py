#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from valid.true_valid_common import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple true-valid run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing metrics/summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "metrics" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_name_from_summary(summary: dict[str, Any]) -> str:
    checkpoint_path = Path(summary["checkpoint_path"])
    parent_name = checkpoint_path.parent.name
    return f"{parent_name}_{checkpoint_path.name}"


def _model_tag_from_summary(summary: dict[str, Any]) -> str:
    model_path = str(summary.get("dorado_model") or "").strip()
    if not model_path:
        return "unknown_model"
    model_name = Path(model_path).name.strip() or "unknown_model"
    return (
        model_name.replace("@", "_at_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_loaded: list[tuple[Path, dict[str, Any]]] = []
    for run_dir_raw in args.run_dir:
        run_dir = Path(run_dir_raw).expanduser().resolve()
        summary = _load_summary(run_dir)
        raw_loaded.append((run_dir, summary))

    base_name_counts: dict[str, int] = {}
    for _, summary in raw_loaded:
        base_name = _run_name_from_summary(summary)
        base_name_counts[base_name] = base_name_counts.get(base_name, 0) + 1

    loaded: list[tuple[str, Path, dict[str, Any]]] = []
    used_names: set[str] = set()
    for run_dir, summary in raw_loaded:
        base_name = _run_name_from_summary(summary)
        if base_name_counts.get(base_name, 0) > 1:
            name = f"{base_name}_{_model_tag_from_summary(summary)}"
        else:
            name = base_name
        suffix = 2
        unique_name = name
        while unique_name in used_names:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)
        loaded.append((unique_name, run_dir, summary))

    compare_payload: dict[str, Any] = {
        "runs": {},
        "ordered_runs": [name for name, _, _ in loaded],
    }
    markdown_lines = [
        "# True Valid Compare",
        "",
        "| Run | Shared | Gen Identity | Gen Empirical Q | Gen FASTQ Q | Delta Identity vs Orig | Delta Empirical Q vs Orig |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, run_dir, summary in loaded:
        generated = dict(summary.get("generated_vs_truth") or {})
        original = dict(summary.get("original_vs_truth") or {})
        compare_payload["runs"][name] = {
            "run_dir": str(run_dir),
            "checkpoint_path": str(summary["checkpoint_path"]),
            "truth_mode": summary.get("truth_mode"),
            "original_vs_truth": original,
            "generated_vs_truth": generated,
            "generated_minus_original_vs_truth": dict(summary.get("generated_minus_original_vs_truth") or {}),
        }
        markdown_lines.append(
            "| {name} | {shared} | {gen_identity:.6f} | {gen_eq:.4f} | {gen_q:.4f} | {delta_identity:+.6f} | {delta_eq:+.4f} |".format(
                name=name,
                shared=int(generated.get("shared_read_count", 0)),
                gen_identity=float(generated.get("length_weighted_identity", 0.0)),
                gen_eq=float(generated.get("mean_empirical_qscore", 0.0)),
                gen_q=float(generated.get("mean_predicted_qscore", 0.0)),
                delta_identity=float(summary.get("generated_minus_original_vs_truth", {}).get("delta_length_weighted_identity", 0.0)),
                delta_eq=float(summary.get("generated_minus_original_vs_truth", {}).get("delta_mean_empirical_qscore", 0.0)),
            )
        )

    write_json(output_dir / "compare_summary.json", compare_payload)
    (output_dir / "RESULTS_COMPARE.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(json.dumps(compare_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
