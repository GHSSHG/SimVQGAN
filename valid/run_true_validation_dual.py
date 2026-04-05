#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_HAC_MODEL = Path("~/Download/dorado/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0").expanduser()
DEFAULT_SUP_MODEL = Path("~/Download/dorado/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0").expanduser()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true-valid for the same checkpoint with HAC and SUP Dorado models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compare-output-dir", type=Path, default=None)
    parser.add_argument("--chunk-batch-size", type=int, default=128)
    parser.add_argument("--recon-mode", type=str, default="overlap_add")
    parser.add_argument("--hop-samples", type=int, default=None)
    parser.add_argument("--trim-mode", type=str, default="drop")
    parser.add_argument("--tail-chunk-mode", type=str, default="drop")
    parser.add_argument("--dorado-bin", type=str, default="dorado")
    parser.add_argument("--dorado-device", type=str, default="cuda:0")
    parser.add_argument("--hac-model", type=Path, default=DEFAULT_HAC_MODEL)
    parser.add_argument("--sup-model", type=Path, default=DEFAULT_SUP_MODEL)
    return parser.parse_args()


def _run_model(
    *,
    label: str,
    dorado_model: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "valid" / "run_true_validation.py"),
        "--config",
        str(args.config.expanduser().resolve()),
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--manifest-path",
        str(args.manifest_path.expanduser().resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--chunk-batch-size",
        str(max(1, int(args.chunk_batch_size))),
        "--recon-mode",
        str(args.recon_mode),
        "--trim-mode",
        str(args.trim_mode),
        "--tail-chunk-mode",
        str(args.tail_chunk_mode),
        "--dorado-bin",
        str(args.dorado_bin),
        "--dorado-model",
        str(dorado_model.expanduser().resolve()),
        "--dorado-device",
        str(args.dorado_device),
    ]
    if args.hop_samples is not None:
        cmd.extend(["--hop-samples", str(int(args.hop_samples))])
    print(f"[dual] running {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    hac_model = args.hac_model.expanduser().resolve()
    sup_model = args.sup_model.expanduser().resolve()
    if not hac_model.exists():
        raise FileNotFoundError(f"HAC model not found: {hac_model}")
    if not sup_model.exists():
        raise FileNotFoundError(f"SUP model not found: {sup_model}")

    hac_output_dir = output_root / "hac"
    sup_output_dir = output_root / "sup"
    _run_model(label="hac", dorado_model=hac_model, args=args, output_dir=hac_output_dir)
    _run_model(label="sup", dorado_model=sup_model, args=args, output_dir=sup_output_dir)

    compare_output_dir = (
        args.compare_output_dir.expanduser().resolve()
        if args.compare_output_dir is not None
        else (output_root / "compare_hac_vs_sup").resolve()
    )
    compare_output_dir.mkdir(parents=True, exist_ok=True)
    compare_cmd = [
        sys.executable,
        str(REPO_ROOT / "valid" / "compare_true_valid_runs.py"),
        "--run-dir",
        str(hac_output_dir.resolve()),
        "--run-dir",
        str(sup_output_dir.resolve()),
        "--output-dir",
        str(compare_output_dir),
    ]
    print(f"[dual] comparing hac vs sup: {' '.join(compare_cmd)}", flush=True)
    subprocess.run(compare_cmd, check=True, cwd=str(REPO_ROOT))

    payload = {
        "status": "ok",
        "output_root": str(output_root),
        "runs": {
            "hac": str(hac_output_dir),
            "sup": str(sup_output_dir),
        },
        "compare_output_dir": str(compare_output_dir),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
